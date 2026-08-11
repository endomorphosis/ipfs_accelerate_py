"""Canonical contracts for incremental verification.

This module is a serialization and identity boundary, not an execution or
cache-authority boundary.  It deliberately reuses the supervisor's canonical
DAG-JSON and CID profile.  A structurally valid receipt still requires the
executor/cache admission checks which observed its inputs; a CID, provider
claim, or signature does not manufacture verification authority.

Public records contain compact values and content references only.  Raw logs,
environment variables, credentials, proof witnesses, and other private
material must remain in separately protected artifact storage.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, TypeAlias, TypeVar

from ..core.multiformats_identity import (
    MultiformatsIdentityError,
    cid_for_bytes,
    cid_for_dag_json,
    validate_cid,
)
from ..proof.formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    EvidenceFreshness,
    ProofVerdict,
    canonical_json_bytes,
    content_identity,
)
from ..proof.formal_verification_contracts import (
    ProofReceipt as FormalProofReceipt,
)
from ..proof.test_execution_contracts import TestExecutionKey, TestPassReceipt

VERIFICATION_CONTRACT_VERSION: Final[int] = 1
MAX_TEXT_BYTES: Final[int] = 8_192
MAX_REASON_BYTES: Final[int] = 512
MAX_COLLECTION_ITEMS: Final[int] = 512
MAX_MAPPING_ITEMS: Final[int] = 512
MAX_CANONICAL_DEPTH: Final[int] = 16
MAX_CANONICAL_ITEMS: Final[int] = 4_096
MAX_RECORD_BYTES: Final[int] = 1_048_576
MAX_SUMMARY_BYTES: Final[int] = 262_144
MAX_COUNTEREXAMPLE_BYTES: Final[int] = 262_144
MAX_RAW_IDENTITY_BYTES: Final[int] = 16 * 1_048_576
MAX_DURATION_MS: Final[int] = 7 * 24 * 60 * 60 * 1_000
MAX_RESOURCE_QUANTITY: Final[int] = 2**63 - 1

VERIFICATION_RECEIPT_KEY_INTERFACE: Final[str] = "VerificationReceiptKey@1"
DIRECT_EXECUTION_OBSERVATION_INTERFACE: Final[str] = "DirectExecutionObservation@1"
STATIC_ANALYSIS_RECEIPT_INTERFACE: Final[str] = "StaticAnalysisReceipt@1"
TYPE_CHECK_RECEIPT_INTERFACE: Final[str] = "TypeCheckReceipt@1"
TEST_RECEIPT_INTERFACE: Final[str] = "TestReceipt@1"
PROOF_RECEIPT_INTERFACE: Final[str] = "ProofReceipt@1"
COUNTEREXAMPLE_RECEIPT_INTERFACE: Final[str] = "CounterexampleReceipt@1"
VERIFICATION_PLAN_INTERFACE: Final[str] = "VerificationPlan@1"
VERIFICATION_BUNDLE_INTERFACE: Final[str] = "VerificationBundle@1"
VERIFICATION_SUMMARY_INTERFACE: Final[str] = "VerificationSummary@1"
CACHE_REUSE_DECISION_INTERFACE: Final[str] = "CacheReuseDecision@1"
MODEL_ROUTE_DECISION_INTERFACE: Final[str] = "ModelRouteDecision@1"
VERIFICATION_COMMITMENT_INTERFACE: Final[str] = "VerificationCommitment@1"

VERIFICATION_RECEIPT_KEY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-receipt-key@1"
)
DIRECT_EXECUTION_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/direct-verification-observation@1"
)
STATIC_ANALYSIS_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/static-analysis-receipt@1"
)
TYPE_CHECK_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/type-check-receipt@1"
)
TEST_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-test-receipt@1"
)
PROOF_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-proof-receipt@1"
)
COUNTEREXAMPLE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-counterexample-receipt@1"
)
VERIFICATION_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-plan@1"
)
VERIFICATION_BUNDLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-bundle@1"
)
VERIFICATION_SUMMARY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-summary@1"
)
CACHE_REUSE_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/cache-reuse-decision@1"
)
MODEL_ROUTE_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/model-route-decision@1"
)
VERIFICATION_COMMITMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-commitment@1"
)

_TREE_IDENTITY_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/observed-repository-tree@1"
)
_SEMANTIC_IDENTITY_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/observed-semantic-state@1"
)
_SYMBOL_IDENTITY_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/affected-symbol-version@1"
)
_ENVIRONMENT_IDENTITY_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/effective-verification-environment@1"
)
_SELECTOR_IDENTITY_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-selector-argv@1"
)
_OBLIGATION_IDENTITY_INPUT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/verification-proof-obligation@1"
)
_OBLIGATION_NOT_APPLICABLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/proof-obligation-not-applicable@1"
)
_ABSENT_BYTES_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/absent-verification-bytes@1"
)

_VERSIONED_SCHEMA_RE: Final[re.Pattern[str]] = re.compile(r"^[^\x00\r\n]{1,512}@[1-9][0-9]*$")
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_PRIVATE_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "refresh_token",
        "secret",
        "session_token",
        "witness",
    }
)


class VerificationContractError(ContractValidationError):
    """A verification record is malformed or unsafe."""


class VerificationBoundsError(VerificationContractError):
    """A verification record exceeds a closed public bound."""


class VerificationIdentityError(VerificationContractError):
    """Observed identity material and a claimed identity disagree."""


class TerminalStatus(str, Enum):
    """Closed terminal status vocabulary for all verification receipts."""

    PASSED = "passed"
    FAILED = "failed"
    PROVED = "proved"
    DISPROVED = "disproved"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    NOT_MODELED = "not_modeled"
    STALE = "stale"
    INVALID = "invalid"
    CANCELLED = "cancelled"
    SIMULATED = "simulated"

    @property
    def terminal(self) -> bool:
        return True

    @property
    def successful(self) -> bool:
        return self in {TerminalStatus.PASSED, TerminalStatus.PROVED}


class VerificationReceiptKind(str, Enum):
    STATIC_ANALYSIS = "static_analysis"
    TYPE_CHECK = "type_check"
    TEST = "test"
    PROOF = "proof"


class CacheReuseDisposition(str, Enum):
    REUSED = "reused"
    STALE = "stale"
    MISSING = "missing"
    CORRUPT = "corrupt"
    MISMATCHED = "mismatched"
    SIMULATED = "simulated"
    NON_AUTHORITATIVE = "non_authoritative"
    POLICY_REJECTED = "policy_rejected"
    TERMINAL_STATUS_REJECTED = "terminal_status_rejected"


class ModelRoute(str, Enum):
    DETERMINISTIC_ONLY = "deterministic_only"
    SMALL_LOCAL_MODEL = "small_local_model"
    MEDIUM_MODEL = "medium_model"
    FRONTIER_MODEL = "frontier_model"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


class DiagnosticValueState(str, Enum):
    PRESENT = "present"
    REDACTED = "redacted"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


TEnum = TypeVar("TEnum", bound=Enum)
TContract = TypeVar("TContract", bound=CanonicalContract)


def _enum(value: Any, enum_type: type[TEnum], *, field_name: str) -> TEnum:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(raw)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise VerificationContractError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
) -> str:
    if not isinstance(value, str):
        raise VerificationContractError(f"{field_name} must be a string")
    result = value.strip()
    if required and not result:
        raise VerificationContractError(f"{field_name} must not be empty")
    if "\x00" in result:
        raise VerificationContractError(f"{field_name} must not contain NUL")
    if len(result.encode("utf-8")) > maximum:
        raise VerificationBoundsError(
            f"{field_name} exceeds {maximum} UTF-8 bytes"
        )
    return result


def _token(value: Any, *, field_name: str) -> str:
    result = _text(value, field_name=field_name, maximum=128)
    if not _TOKEN_RE.fullmatch(result):
        raise VerificationContractError(f"{field_name} is not a canonical token")
    return result


def _versioned_schema(value: Any, *, field_name: str) -> str:
    result = _text(value, field_name=field_name, maximum=512)
    if not _VERSIONED_SCHEMA_RE.fullmatch(result):
        raise VerificationContractError(
            f"{field_name} must be an explicitly versioned @N schema"
        )
    return result


def _integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int = MAX_RESOURCE_QUANTITY,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerificationContractError(f"{field_name} must be an integer")
    if value < minimum or value > maximum:
        raise VerificationBoundsError(
            f"{field_name} must be between {minimum} and {maximum}"
        )
    return value


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise VerificationContractError(f"{field_name} must be a boolean")
    return value


def _cid(value: Any, *, field_name: str, required: bool = True) -> str:
    result = _text(value, field_name=field_name, required=required, maximum=256)
    if not result:
        return ""
    try:
        return validate_cid(result, codecs=("raw", "dag-json"))
    except MultiformatsIdentityError as exc:
        raise VerificationIdentityError(
            f"{field_name} must use the frozen CIDv1 profile"
        ) from exc


def _sha256(value: Any, *, field_name: str) -> str:
    result = _text(value, field_name=field_name, maximum=71)
    if not _SHA256_RE.fullmatch(result):
        raise VerificationIdentityError(
            f"{field_name} must be sha256 followed by 64 lowercase hex characters"
        )
    return result


def _private_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(
        normalized == marker
        or normalized.endswith("_" + marker)
        or marker in normalized
        for marker in _PRIVATE_FIELD_MARKERS
    )


def _freeze_public(
    value: Any,
    *,
    field_name: str,
    depth: int = 0,
    budget: list[int] | None = None,
    active: set[int] | None = None,
) -> Any:
    """Deep-freeze one bounded canonical public value and reject secrets."""

    if budget is None:
        budget = [0]
    if active is None:
        active = set()
    if depth > MAX_CANONICAL_DEPTH:
        raise VerificationBoundsError(f"{field_name} exceeds canonical depth")
    budget[0] += 1
    if budget[0] > MAX_CANONICAL_ITEMS:
        raise VerificationBoundsError(f"{field_name} exceeds canonical item bound")

    if value is None or type(value) in {bool, int}:
        return value
    if isinstance(value, str):
        return _text(value, field_name=field_name, required=False)
    if isinstance(value, float):
        raise VerificationContractError(
            f"{field_name} cannot contain floating point values"
        )
    if isinstance(value, Enum):
        return _freeze_public(
            value.value,
            field_name=field_name,
            depth=depth,
            budget=budget,
            active=active,
        )
    if isinstance(value, CanonicalContract):
        return _freeze_public(
            value.to_dict(),
            field_name=field_name,
            depth=depth,
            budget=budget,
            active=active,
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise VerificationContractError(
            f"{field_name} cannot embed raw bytes in a public record"
        )

    container_id = id(value)
    if container_id in active:
        raise VerificationContractError(f"{field_name} cannot contain cycles")
    active.add(container_id)
    try:
        if isinstance(value, Mapping):
            if len(value) > MAX_MAPPING_ITEMS:
                raise VerificationBoundsError(
                    f"{field_name} exceeds mapping item bound"
                )
            frozen: dict[str, Any] = {}
            for raw_key in sorted(value):
                if not isinstance(raw_key, str):
                    raise VerificationContractError(
                        f"{field_name} keys must be strings"
                    )
                if _private_key(raw_key):
                    raise VerificationContractError(
                        f"{field_name} contains private or witness material"
                    )
                key = _text(
                    raw_key,
                    field_name=f"{field_name} key",
                    maximum=512,
                )
                frozen[key] = _freeze_public(
                    value[raw_key],
                    field_name=f"{field_name}.{key}",
                    depth=depth + 1,
                    budget=budget,
                    active=active,
                )
            return MappingProxyType(frozen)
        if isinstance(value, Sequence) and not isinstance(value, str):
            if len(value) > MAX_COLLECTION_ITEMS:
                raise VerificationBoundsError(
                    f"{field_name} exceeds sequence item bound"
                )
            return tuple(
                _freeze_public(
                    item,
                    field_name=f"{field_name}[{index}]",
                    depth=depth + 1,
                    budget=budget,
                    active=active,
                )
                for index, item in enumerate(value)
            )
    finally:
        active.remove(container_id)
    raise VerificationContractError(
        f"{field_name} contains unsupported type {type(value).__name__}"
    )


def _mapping(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerificationContractError(f"{field_name} must be a mapping")
    result = _freeze_public(value, field_name=field_name)
    assert isinstance(result, Mapping)
    if required and not result:
        raise VerificationContractError(f"{field_name} must not be empty")
    return result


def _strings(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
    preserve_order: bool = False,
    maximum: int = MAX_COLLECTION_ITEMS,
    item_bytes: int = MAX_TEXT_BYTES,
) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise VerificationContractError(f"{field_name} must be a sequence")
    if len(values) > maximum:
        raise VerificationBoundsError(f"{field_name} exceeds {maximum} items")
    result: list[str] = []
    for index, value in enumerate(values):
        item = _text(
            value,
            field_name=f"{field_name}[{index}]",
            maximum=item_bytes,
        )
        if item in result:
            raise VerificationContractError(
                f"{field_name} must not contain duplicates"
            )
        result.append(item)
    if required and not result:
        raise VerificationContractError(f"{field_name} must not be empty")
    return tuple(result if preserve_order else sorted(result))


def _cids(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
    preserve_order: bool = False,
    maximum: int = MAX_COLLECTION_ITEMS,
) -> tuple[str, ...]:
    raw = _strings(
        values,
        field_name=field_name,
        required=required,
        preserve_order=True,
        maximum=maximum,
        item_bytes=256,
    )
    result = tuple(_cid(item, field_name=field_name) for item in raw)
    return result if preserve_order else tuple(sorted(result))


def _reason_codes(values: Any, *, field_name: str, required: bool = False) -> tuple[str, ...]:
    result = _strings(
        values,
        field_name=field_name,
        required=required,
        maximum=128,
        item_bytes=MAX_REASON_BYTES,
    )
    for value in result:
        if not _TOKEN_RE.fullmatch(value):
            raise VerificationContractError(
                f"{field_name} entries must be canonical reason tokens"
            )
    return result


def _check_header(
    payload: Mapping[str, Any],
    *,
    schema: str,
    interface: str,
    artifact_name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise VerificationContractError(f"{artifact_name} must be an object")
    if payload.get("schema") != schema:
        raise VerificationContractError(f"{artifact_name} has an unsupported schema")
    version = payload.get("contract_version")
    if (
        type(version) is not int
        or version != VERIFICATION_CONTRACT_VERSION
    ):
        raise VerificationContractError(
            f"{artifact_name} has an unsupported contract version"
        )
    if payload.get("interface") != interface:
        raise VerificationContractError(
            f"{artifact_name} has an unsupported interface"
        )


def _reject_unknown(
    payload: Mapping[str, Any],
    fields: Iterable[str],
    *,
    artifact_name: str,
) -> None:
    allowed = set(fields) | {
        "schema",
        "contract_version",
        "interface",
        "content_id",
    }
    if set(payload).difference(allowed):
        raise VerificationContractError(
            f"{artifact_name} contains unsupported fields"
        )


def _check_identity(
    payload: Mapping[str, Any],
    actual: str,
    *,
    names: Sequence[str],
    artifact_name: str,
) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise VerificationIdentityError(
                f"{artifact_name} content identity does not match payload"
            )


def _check_projection(
    payload: Mapping[str, Any],
    *,
    field_name: str,
    actual: Any,
) -> None:
    if field_name not in payload:
        return
    def normalized(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Mapping):
            return {
                str(key): normalized(item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return [normalized(item) for item in value]
        return value

    expected = normalized(actual)
    raw = normalized(payload[field_name])
    if isinstance(expected, bool) and type(raw) is not bool:
        raise VerificationIdentityError(
            f"{field_name} does not match its derived projection"
        )
    if (
        isinstance(expected, int)
        and not isinstance(expected, bool)
        and type(raw) is not int
    ):
        raise VerificationIdentityError(
            f"{field_name} does not match its derived projection"
        )
    if raw != expected:
        raise VerificationIdentityError(
            f"{field_name} does not match its derived projection"
        )


def _bounded(
    value: CanonicalContract,
    *,
    artifact_name: str,
    maximum: int = MAX_RECORD_BYTES,
) -> None:
    try:
        encoded = value.canonical_bytes()
    except ContractValidationError:
        raise
    except Exception as exc:
        raise VerificationContractError(
            f"{artifact_name} is not canonical"
        ) from exc
    if len(encoded) > maximum:
        raise VerificationBoundsError(
            f"{artifact_name} exceeds {maximum} canonical bytes"
        )


def _record(
    value: Any,
    record_type: type[TContract],
    *,
    field_name: str,
    optional: bool = False,
) -> TContract | None:
    if value is None and optional:
        return None
    if isinstance(value, record_type):
        return value
    if isinstance(value, Mapping):
        decoder = record_type.from_dict
        return decoder(value)
    raise VerificationContractError(
        f"{field_name} must be a {record_type.__name__} record"
    )


def _structured_cid(schema: str, value: Any, *, field_name: str) -> str:
    public_value = _freeze_public(value, field_name=field_name)
    envelope = {"schema": schema, "value": public_value}
    # The formal encoder rejects floats and unsupported values.  Decode its
    # exact bytes before exercising the independent multiformats entry point.
    encoded = canonical_json_bytes(envelope)
    decoded = json.loads(encoded.decode("utf-8"))
    formal_cid = content_identity(decoded)
    try:
        multiformats_cid = cid_for_dag_json(decoded, for_identity=True)
    except MultiformatsIdentityError as exc:
        raise VerificationIdentityError(
            f"{field_name} cannot be represented by the frozen identity profile"
        ) from exc
    if formal_cid != multiformats_cid:
        raise VerificationIdentityError(
            f"{field_name} disagrees across canonical identity implementations"
        )
    return _cid(formal_cid, field_name=field_name)


def _bytes_cid(value: bytes | None, *, field_name: str) -> str:
    if value is None:
        return _structured_cid(
            _ABSENT_BYTES_IDENTITY_SCHEMA,
            {"field": field_name, "state": "not_present"},
            field_name=field_name,
        )
    if type(value) is not bytes:
        raise VerificationContractError(f"{field_name} must be exact bytes or None")
    if len(value) > MAX_RAW_IDENTITY_BYTES:
        raise VerificationBoundsError(
            f"{field_name} exceeds {MAX_RAW_IDENTITY_BYTES} bytes"
        )
    try:
        return validate_cid(cid_for_bytes(value), codecs=("raw",))
    except MultiformatsIdentityError as exc:
        raise VerificationIdentityError(
            f"{field_name} cannot be content addressed"
        ) from exc


PROOF_OBLIGATION_NOT_APPLICABLE_CID: Final[str] = _structured_cid(
    _OBLIGATION_NOT_APPLICABLE_SCHEMA,
    {"state": "not_applicable", "reason": "non_proof_receipt_kind"},
    field_name="proof_obligation_not_applicable",
)


class _VerificationContract(CanonicalContract):
    INTERFACE: ClassVar[str] = ""

    @property
    def interface(self) -> str:
        return self.INTERFACE

    @property
    def schema_version(self) -> int:
        return VERIFICATION_CONTRACT_VERSION

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "content_id": self.content_id}


@dataclass(frozen=True)
class VerificationReceiptKey(_VerificationContract):
    """Exact cache key over every verification-authority input."""

    SCHEMA: ClassVar[str] = VERIFICATION_RECEIPT_KEY_SCHEMA
    INTERFACE: ClassVar[str] = VERIFICATION_RECEIPT_KEY_INTERFACE

    repository_tree_cid: str
    semantic_state_root_cid: str
    affected_symbol_version_cids: tuple[str, ...]
    environment_cid: str
    dependency_lock_cid: str
    selector_cid: str
    proof_obligation_cid: str
    tool_name: str
    tool_version: str
    configuration_cid: str
    fixture_data_cids: tuple[str, ...]
    network_policy: str
    receipt_schema_version: int
    receipt_kind: VerificationReceiptKind
    adapter_schema: str

    def __post_init__(self) -> None:
        for name in (
            "repository_tree_cid",
            "semantic_state_root_cid",
            "environment_cid",
            "dependency_lock_cid",
            "selector_cid",
            "proof_obligation_cid",
            "configuration_cid",
        ):
            object.__setattr__(self, name, _cid(getattr(self, name), field_name=name))
        object.__setattr__(
            self,
            "affected_symbol_version_cids",
            _cids(
                self.affected_symbol_version_cids,
                field_name="affected_symbol_version_cids",
            ),
        )
        object.__setattr__(
            self,
            "fixture_data_cids",
            _cids(self.fixture_data_cids, field_name="fixture_data_cids"),
        )
        object.__setattr__(
            self, "tool_name", _text(self.tool_name, field_name="tool_name", maximum=256)
        )
        object.__setattr__(
            self,
            "tool_version",
            _text(self.tool_version, field_name="tool_version", maximum=256),
        )
        object.__setattr__(
            self, "network_policy", _token(self.network_policy, field_name="network_policy")
        )
        object.__setattr__(
            self,
            "receipt_schema_version",
            _integer(
                self.receipt_schema_version,
                field_name="receipt_schema_version",
                minimum=1,
                maximum=2**31 - 1,
            ),
        )
        object.__setattr__(
            self,
            "receipt_kind",
            _enum(self.receipt_kind, VerificationReceiptKind, field_name="receipt_kind"),
        )
        object.__setattr__(
            self,
            "adapter_schema",
            _versioned_schema(self.adapter_schema, field_name="adapter_schema"),
        )
        if self.receipt_kind is VerificationReceiptKind.PROOF:
            if self.proof_obligation_cid == PROOF_OBLIGATION_NOT_APPLICABLE_CID:
                raise VerificationIdentityError(
                    "proof receipts require an applicable proof obligation"
                )
        elif self.proof_obligation_cid != PROOF_OBLIGATION_NOT_APPLICABLE_CID:
            raise VerificationIdentityError(
                "non-proof receipts require the canonical not-applicable obligation"
            )
        _bounded(self, artifact_name="verification receipt key")

    @property
    def key_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "repository_tree_cid": self.repository_tree_cid,
            "semantic_state_root_cid": self.semantic_state_root_cid,
            "affected_symbol_version_cids": self.affected_symbol_version_cids,
            "environment_cid": self.environment_cid,
            "dependency_lock_cid": self.dependency_lock_cid,
            "selector_cid": self.selector_cid,
            "proof_obligation_cid": self.proof_obligation_cid,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "configuration_cid": self.configuration_cid,
            "fixture_data_cids": self.fixture_data_cids,
            "network_policy": self.network_policy,
            "receipt_schema_version": self.receipt_schema_version,
            "receipt_kind": self.receipt_kind,
            "adapter_schema": self.adapter_schema,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "key_id": self.key_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationReceiptKey:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="verification receipt key",
        )
        fields = {
            "repository_tree_cid",
            "semantic_state_root_cid",
            "affected_symbol_version_cids",
            "environment_cid",
            "dependency_lock_cid",
            "selector_cid",
            "proof_obligation_cid",
            "tool_name",
            "tool_version",
            "configuration_cid",
            "fixture_data_cids",
            "network_policy",
            "receipt_schema_version",
            "receipt_kind",
            "adapter_schema",
        }
        _reject_unknown(
            payload,
            fields | {"key_id"},
            artifact_name="verification receipt key",
        )
        result = cls(
            repository_tree_cid=payload.get("repository_tree_cid", ""),
            semantic_state_root_cid=payload.get("semantic_state_root_cid", ""),
            affected_symbol_version_cids=tuple(
                payload.get("affected_symbol_version_cids") or ()
            ),
            environment_cid=payload.get("environment_cid", ""),
            dependency_lock_cid=payload.get("dependency_lock_cid", ""),
            selector_cid=payload.get("selector_cid", ""),
            proof_obligation_cid=payload.get("proof_obligation_cid", ""),
            tool_name=payload.get("tool_name", ""),
            tool_version=payload.get("tool_version", ""),
            configuration_cid=payload.get("configuration_cid", ""),
            fixture_data_cids=tuple(payload.get("fixture_data_cids") or ()),
            network_policy=payload.get("network_policy", ""),
            receipt_schema_version=payload.get("receipt_schema_version", 0),
            receipt_kind=payload.get("receipt_kind", ""),
            adapter_schema=payload.get("adapter_schema", ""),
        )
        _check_identity(
            payload,
            result.key_id,
            names=("key_id", "content_id"),
            artifact_name="verification receipt key",
        )
        return result


@dataclass(frozen=True)
class VerificationIdentityCompiler:
    """Compile exact keys from observed values and cross-check caller claims.

    The compiler is deliberately pure.  Runtime adapters are responsible for
    observing the filesystem, executable, sandbox, and tool versions; this
    class refuses to accept precomputed component overrides in their place.
    """

    def compile_key(
        self,
        *,
        observed_repository_tree: Mapping[str, Any],
        claimed_repository_tree_cid: str,
        patch_base_tree_id: str,
        repository_state_tree_id: str,
        invalidation_plan_tree_id: str,
        context_pack_tree_id: str,
        observed_semantic_state: Mapping[str, Any],
        repository_state_semantic_root_cid: str,
        invalidation_plan_semantic_root_cid: str,
        context_pack_semantic_root_cid: str,
        affected_symbol_versions: Sequence[Mapping[str, Any]],
        observed_environment: Mapping[str, Any],
        claimed_environment_cid: str,
        dependency_lock_bytes: bytes | None,
        selector_argv: Sequence[str],
        proof_obligation: Mapping[str, Any] | None,
        tool_name: str,
        tool_version: str,
        configuration_bytes: bytes | None,
        fixture_data_bytes: Sequence[bytes],
        network_policy: str,
        receipt_schema_version: int,
        receipt_kind: VerificationReceiptKind | str,
        adapter_schema: str,
    ) -> VerificationReceiptKey:
        base_ids = tuple(
            _text(value, field_name=name, maximum=512)
            for name, value in (
                ("patch_base_tree_id", patch_base_tree_id),
                ("repository_state_tree_id", repository_state_tree_id),
                ("invalidation_plan_tree_id", invalidation_plan_tree_id),
                ("context_pack_tree_id", context_pack_tree_id),
            )
        )
        if len(set(base_ids)) != 1:
            raise VerificationIdentityError(
                "patch, repository, invalidation, and context base trees disagree"
            )

        tree_cid = _structured_cid(
            _TREE_IDENTITY_INPUT_SCHEMA,
            _mapping(
                observed_repository_tree,
                field_name="observed_repository_tree",
                required=True,
            ),
            field_name="observed_repository_tree",
        )
        claimed_tree = _cid(
            claimed_repository_tree_cid,
            field_name="claimed_repository_tree_cid",
        )
        if tree_cid != claimed_tree:
            raise VerificationIdentityError(
                "claimed repository tree CID does not match observed patched tree"
            )

        semantic_cid = _structured_cid(
            _SEMANTIC_IDENTITY_INPUT_SCHEMA,
            _mapping(
                observed_semantic_state,
                field_name="observed_semantic_state",
                required=True,
            ),
            field_name="observed_semantic_state",
        )
        semantic_claims = tuple(
            _cid(value, field_name=name)
            for name, value in (
                (
                    "repository_state_semantic_root_cid",
                    repository_state_semantic_root_cid,
                ),
                (
                    "invalidation_plan_semantic_root_cid",
                    invalidation_plan_semantic_root_cid,
                ),
                ("context_pack_semantic_root_cid", context_pack_semantic_root_cid),
            )
        )
        if any(value != semantic_cid for value in semantic_claims):
            raise VerificationIdentityError(
                "repository, invalidation, context, and observed semantic roots disagree"
            )

        policy = _token(network_policy, field_name="network_policy")
        environment = _mapping(
            observed_environment,
            field_name="observed_environment",
            required=True,
        )
        if environment.get("network_policy") != policy:
            raise VerificationIdentityError(
                "effective environment network policy does not match key policy"
            )
        environment_cid = _structured_cid(
            _ENVIRONMENT_IDENTITY_INPUT_SCHEMA,
            environment,
            field_name="observed_environment",
        )
        if environment_cid != _cid(
            claimed_environment_cid, field_name="claimed_environment_cid"
        ):
            raise VerificationIdentityError(
                "claimed environment CID does not match effective environment"
            )

        if isinstance(affected_symbol_versions, (str, bytes)) or not isinstance(
            affected_symbol_versions, Sequence
        ):
            raise VerificationContractError(
                "affected_symbol_versions must be a sequence of mappings"
            )
        if len(affected_symbol_versions) > MAX_COLLECTION_ITEMS:
            raise VerificationBoundsError(
                "affected_symbol_versions exceeds item bound"
            )
        symbol_cids = tuple(
            _structured_cid(
                _SYMBOL_IDENTITY_INPUT_SCHEMA,
                _mapping(
                    item,
                    field_name=f"affected_symbol_versions[{index}]",
                    required=True,
                ),
                field_name=f"affected_symbol_versions[{index}]",
            )
            for index, item in enumerate(affected_symbol_versions)
        )
        if len(symbol_cids) != len(set(symbol_cids)):
            raise VerificationContractError(
                "affected_symbol_versions contains duplicate identities"
            )

        selector = _strings(
            selector_argv,
            field_name="selector_argv",
            required=True,
            preserve_order=True,
        )
        selector_cid = _structured_cid(
            _SELECTOR_IDENTITY_INPUT_SCHEMA,
            {"argv": selector},
            field_name="selector_argv",
        )

        kind = _enum(receipt_kind, VerificationReceiptKind, field_name="receipt_kind")
        if kind is VerificationReceiptKind.PROOF:
            if proof_obligation is None:
                raise VerificationIdentityError(
                    "proof receipts require a normalized obligation translation"
                )
            obligation = _mapping(
                proof_obligation,
                field_name="proof_obligation",
                required=True,
            )
            required_obligation_fields = {
                "normalized_obligation",
                "translation_scheme",
                "negation_scheme",
                "translator_version",
            }
            if not required_obligation_fields.issubset(obligation):
                raise VerificationIdentityError(
                    "proof obligation lacks normalized translation bindings"
                )
            proof_obligation_cid = _structured_cid(
                _OBLIGATION_IDENTITY_INPUT_SCHEMA,
                obligation,
                field_name="proof_obligation",
            )
        else:
            if proof_obligation is not None:
                raise VerificationIdentityError(
                    "non-proof receipts cannot carry a proof obligation"
                )
            proof_obligation_cid = PROOF_OBLIGATION_NOT_APPLICABLE_CID

        if isinstance(fixture_data_bytes, (str, bytes)) or not isinstance(
            fixture_data_bytes, Sequence
        ):
            raise VerificationContractError(
                "fixture_data_bytes must be a sequence of exact bytes"
            )
        if len(fixture_data_bytes) > MAX_COLLECTION_ITEMS:
            raise VerificationBoundsError("fixture_data_bytes exceeds item bound")
        fixture_cids = tuple(
            _bytes_cid(value, field_name=f"fixture_data_bytes[{index}]")
            for index, value in enumerate(fixture_data_bytes)
        )
        if len(fixture_cids) != len(set(fixture_cids)):
            raise VerificationContractError(
                "fixture_data_bytes contains duplicate identities"
            )

        return VerificationReceiptKey(
            repository_tree_cid=tree_cid,
            semantic_state_root_cid=semantic_cid,
            affected_symbol_version_cids=tuple(sorted(symbol_cids)),
            environment_cid=environment_cid,
            dependency_lock_cid=_bytes_cid(
                dependency_lock_bytes, field_name="dependency_lock_bytes"
            ),
            selector_cid=selector_cid,
            proof_obligation_cid=proof_obligation_cid,
            tool_name=_text(tool_name, field_name="tool_name", maximum=256),
            tool_version=_text(tool_version, field_name="tool_version", maximum=256),
            configuration_cid=_bytes_cid(
                configuration_bytes, field_name="configuration_bytes"
            ),
            fixture_data_cids=tuple(sorted(fixture_cids)),
            network_policy=policy,
            receipt_schema_version=receipt_schema_version,
            receipt_kind=kind,
            adapter_schema=adapter_schema,
        )


@dataclass(frozen=True)
class DirectExecutionObservation(_VerificationContract):
    """One current direct tool observation bound to an exact receipt key.

    This is structural evidence only.  Construction does not prove that the
    command ran; the admitted process runner owns that authority boundary.
    """

    SCHEMA: ClassVar[str] = DIRECT_EXECUTION_OBSERVATION_SCHEMA
    INTERFACE: ClassVar[str] = DIRECT_EXECUTION_OBSERVATION_INTERFACE

    receipt_key_cid: str
    repository_tree_cid: str
    environment_cid: str
    terminal_status: TerminalStatus
    command_argv: tuple[str, ...]
    duration_ms: int
    exit_code: int | None = None
    stdout_artifact_cid: str = ""
    stderr_artifact_cid: str = ""
    artifact_cids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("receipt_key_cid", "repository_tree_cid", "environment_cid"):
            object.__setattr__(self, name, _cid(getattr(self, name), field_name=name))
        object.__setattr__(
            self,
            "terminal_status",
            _enum(self.terminal_status, TerminalStatus, field_name="terminal_status"),
        )
        object.__setattr__(
            self,
            "command_argv",
            _strings(
                self.command_argv,
                field_name="command_argv",
                required=True,
                preserve_order=True,
            ),
        )
        object.__setattr__(
            self,
            "duration_ms",
            _integer(
                self.duration_ms,
                field_name="duration_ms",
                maximum=MAX_DURATION_MS,
            ),
        )
        if self.exit_code is not None:
            object.__setattr__(
                self,
                "exit_code",
                _integer(
                    self.exit_code,
                    field_name="exit_code",
                    minimum=-(2**31),
                    maximum=2**31 - 1,
                ),
            )
        for name in ("stdout_artifact_cid", "stderr_artifact_cid"):
            object.__setattr__(
                self,
                name,
                _cid(getattr(self, name), field_name=name, required=False),
            )
        object.__setattr__(
            self,
            "artifact_cids",
            _cids(self.artifact_cids, field_name="artifact_cids"),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _reason_codes(self.reason_codes, field_name="reason_codes"),
        )
        _bounded(self, artifact_name="direct execution observation")

    @property
    def observation_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "receipt_key_cid": self.receipt_key_cid,
            "repository_tree_cid": self.repository_tree_cid,
            "environment_cid": self.environment_cid,
            "terminal_status": self.terminal_status,
            "command_argv": self.command_argv,
            "duration_ms": self.duration_ms,
            "exit_code": self.exit_code,
            "stdout_artifact_cid": self.stdout_artifact_cid,
            "stderr_artifact_cid": self.stderr_artifact_cid,
            "artifact_cids": self.artifact_cids,
            "reason_codes": self.reason_codes,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "observation_id": self.observation_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DirectExecutionObservation:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="direct execution observation",
        )
        fields = {
            "receipt_key_cid",
            "repository_tree_cid",
            "environment_cid",
            "terminal_status",
            "command_argv",
            "duration_ms",
            "exit_code",
            "stdout_artifact_cid",
            "stderr_artifact_cid",
            "artifact_cids",
            "reason_codes",
            "observation_id",
        }
        _reject_unknown(
            payload, fields, artifact_name="direct execution observation"
        )
        result = cls(
            receipt_key_cid=payload.get("receipt_key_cid", ""),
            repository_tree_cid=payload.get("repository_tree_cid", ""),
            environment_cid=payload.get("environment_cid", ""),
            terminal_status=payload.get("terminal_status", ""),
            command_argv=tuple(payload.get("command_argv") or ()),
            duration_ms=payload.get("duration_ms", -1),
            exit_code=payload.get("exit_code"),
            stdout_artifact_cid=payload.get("stdout_artifact_cid", ""),
            stderr_artifact_cid=payload.get("stderr_artifact_cid", ""),
            artifact_cids=tuple(payload.get("artifact_cids") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        _check_identity(
            payload,
            result.observation_id,
            names=("observation_id", "content_id"),
            artifact_name="direct execution observation",
        )
        return result


def _key(value: Any, *, field_name: str = "key") -> VerificationReceiptKey:
    result = _record(value, VerificationReceiptKey, field_name=field_name)
    assert isinstance(result, VerificationReceiptKey)
    return result


def _observation(
    value: Any, *, field_name: str = "execution"
) -> DirectExecutionObservation:
    result = _record(value, DirectExecutionObservation, field_name=field_name)
    assert isinstance(result, DirectExecutionObservation)
    return result


def _validate_execution_binding(
    key: VerificationReceiptKey,
    execution: DirectExecutionObservation,
) -> None:
    if execution.receipt_key_cid != key.key_id:
        raise VerificationIdentityError(
            "execution observation does not bind the receipt key"
        )
    if execution.repository_tree_cid != key.repository_tree_cid:
        raise VerificationIdentityError(
            "execution observation uses a different repository tree"
        )
    if execution.environment_cid != key.environment_cid:
        raise VerificationIdentityError(
            "execution observation uses a different environment"
        )


def _validate_receipt_kind(
    key: VerificationReceiptKey,
    expected: VerificationReceiptKind,
) -> None:
    if key.receipt_kind is not expected:
        raise VerificationContractError(
            f"receipt requires key kind {expected.value}"
        )


def _receipt_artifacts(values: Any) -> tuple[str, ...]:
    return _cids(values, field_name="artifact_cids")


def _direct_check_status(status: TerminalStatus, *, proof: bool = False) -> None:
    disallowed = {TerminalStatus.PROVED, TerminalStatus.DISPROVED}
    if proof:
        disallowed |= {TerminalStatus.PASSED, TerminalStatus.FAILED}
    if status in disallowed:
        raise VerificationContractError(
            "conclusive proof statuses must derive from authoritative proof evidence"
            if proof
            else "non-proof execution cannot use proof terminal statuses"
        )


@dataclass(frozen=True)
class StaticAnalysisReceipt(_VerificationContract):
    SCHEMA: ClassVar[str] = STATIC_ANALYSIS_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = STATIC_ANALYSIS_RECEIPT_INTERFACE

    key: VerificationReceiptKey
    execution: DirectExecutionObservation
    artifact_cids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _key(self.key))
        object.__setattr__(self, "execution", _observation(self.execution))
        _validate_receipt_kind(self.key, VerificationReceiptKind.STATIC_ANALYSIS)
        _validate_execution_binding(self.key, self.execution)
        _direct_check_status(self.execution.terminal_status)
        object.__setattr__(self, "artifact_cids", _receipt_artifacts(self.artifact_cids))
        object.__setattr__(
            self, "reason_codes", _reason_codes(self.reason_codes, field_name="reason_codes")
        )
        _bounded(self, artifact_name="static analysis receipt")

    @property
    def status(self) -> TerminalStatus:
        return self.execution.terminal_status

    @property
    def terminal_success(self) -> bool:
        return self.status is TerminalStatus.PASSED

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "key": self.key.to_record(),
            "execution": self.execution.to_record(),
            "status": self.status,
            "artifact_cids": self.artifact_cids,
            "reason_codes": self.reason_codes,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StaticAnalysisReceipt:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="static analysis receipt",
        )
        _reject_unknown(
            payload,
            {"key", "execution", "status", "artifact_cids", "reason_codes", "receipt_id"},
            artifact_name="static analysis receipt",
        )
        result = cls(
            key=_key(payload.get("key")),
            execution=_observation(payload.get("execution")),
            artifact_cids=tuple(payload.get("artifact_cids") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        _check_projection(payload, field_name="status", actual=result.status)
        _check_identity(
            payload,
            result.receipt_id,
            names=("receipt_id", "content_id"),
            artifact_name="static analysis receipt",
        )
        return result


@dataclass(frozen=True)
class TypeCheckReceipt(_VerificationContract):
    SCHEMA: ClassVar[str] = TYPE_CHECK_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = TYPE_CHECK_RECEIPT_INTERFACE

    key: VerificationReceiptKey
    execution: DirectExecutionObservation
    artifact_cids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _key(self.key))
        object.__setattr__(self, "execution", _observation(self.execution))
        _validate_receipt_kind(self.key, VerificationReceiptKind.TYPE_CHECK)
        _validate_execution_binding(self.key, self.execution)
        _direct_check_status(self.execution.terminal_status)
        object.__setattr__(self, "artifact_cids", _receipt_artifacts(self.artifact_cids))
        object.__setattr__(
            self, "reason_codes", _reason_codes(self.reason_codes, field_name="reason_codes")
        )
        _bounded(self, artifact_name="type check receipt")

    @property
    def status(self) -> TerminalStatus:
        return self.execution.terminal_status

    @property
    def terminal_success(self) -> bool:
        return self.status is TerminalStatus.PASSED

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "key": self.key.to_record(),
            "execution": self.execution.to_record(),
            "status": self.status,
            "artifact_cids": self.artifact_cids,
            "reason_codes": self.reason_codes,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TypeCheckReceipt:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="type check receipt",
        )
        _reject_unknown(
            payload,
            {"key", "execution", "status", "artifact_cids", "reason_codes", "receipt_id"},
            artifact_name="type check receipt",
        )
        result = cls(
            key=_key(payload.get("key")),
            execution=_observation(payload.get("execution")),
            artifact_cids=tuple(payload.get("artifact_cids") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        _check_projection(payload, field_name="status", actual=result.status)
        _check_identity(
            payload,
            result.receipt_id,
            names=("receipt_id", "content_id"),
            artifact_name="type check receipt",
        )
        return result


def _test_pass_receipt(value: Any) -> TestPassReceipt:
    if isinstance(value, TestPassReceipt):
        return value
    if isinstance(value, Mapping):
        return TestPassReceipt.from_dict(value)
    raise VerificationContractError(
        "test_pass_receipt must be an existing canonical TestPassReceipt"
    )


def _test_execution_key(value: Any) -> TestExecutionKey:
    if isinstance(value, TestExecutionKey):
        return value
    if isinstance(value, Mapping):
        return TestExecutionKey.from_dict(value)
    raise VerificationContractError(
        "test_execution_key must be an existing canonical TestExecutionKey"
    )


@dataclass(frozen=True)
class TestReceipt(_VerificationContract):
    """Test result whose success is re-derived from direct or existing evidence."""

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = TEST_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = TEST_RECEIPT_INTERFACE

    key: VerificationReceiptKey
    execution: DirectExecutionObservation
    test_pass_receipt: TestPassReceipt | None = None
    test_execution_key: TestExecutionKey | None = None
    artifact_cids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _key(self.key))
        _validate_receipt_kind(self.key, VerificationReceiptKind.TEST)
        execution = _observation(self.execution)
        object.__setattr__(self, "execution", execution)
        _validate_execution_binding(self.key, execution)
        _direct_check_status(execution.terminal_status)
        if (self.test_pass_receipt is None) != (self.test_execution_key is None):
            raise VerificationContractError(
                "existing test bridge requires both TestPassReceipt and TestExecutionKey"
            )
        if self.test_pass_receipt is not None:
            source_key = _test_execution_key(self.test_execution_key)
            object.__setattr__(self, "test_execution_key", source_key)
            source_receipt = _test_pass_receipt(self.test_pass_receipt)
            object.__setattr__(
                self,
                "test_pass_receipt",
                source_receipt,
            )
            if source_receipt.execution_key_cid != source_key.execution_key_id:
                raise VerificationIdentityError(
                    "TestPassReceipt does not bind the supplied TestExecutionKey"
                )
            if source_receipt.locator_cid != source_key.locator_cid:
                raise VerificationIdentityError(
                    "TestPassReceipt locator does not match TestExecutionKey"
                )
            bridge_bindings = {
                "git_tree_id": (source_key.git_tree_id, self.key.repository_tree_cid),
                "environment_cid": (source_key.environment_cid, self.key.environment_cid),
                "dependency_lock_cid": (
                    source_key.dependency_lock_cid,
                    self.key.dependency_lock_cid,
                ),
                "command_semantics_cid": (
                    source_key.command_semantics_cid,
                    self.key.selector_cid,
                ),
                "config_cid": (source_key.config_cid, self.key.configuration_cid),
                "fixture_cids": (
                    tuple(source_key.fixture_cids),
                    tuple(self.key.fixture_data_cids),
                ),
                "pytest_version": (source_key.pytest_version, self.key.tool_version),
            }
            if any(actual != expected for actual, expected in bridge_bindings.values()):
                raise VerificationIdentityError(
                    "existing TestExecutionKey does not match verification receipt key"
                )
            if execution.terminal_status is not TerminalStatus.PASSED:
                raise VerificationIdentityError(
                    "existing passed-test bridge disagrees with direct observation"
                )
        object.__setattr__(self, "artifact_cids", _receipt_artifacts(self.artifact_cids))
        object.__setattr__(
            self, "reason_codes", _reason_codes(self.reason_codes, field_name="reason_codes")
        )
        _bounded(self, artifact_name="test receipt")

    @property
    def status(self) -> TerminalStatus:
        if self.test_pass_receipt is None:
            return self.execution.terminal_status
        if self.test_pass_receipt.admitted and self.test_pass_receipt.all_phases_pass:
            return TerminalStatus.PASSED
        return TerminalStatus.INVALID

    @property
    def terminal_success(self) -> bool:
        return self.status is TerminalStatus.PASSED

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "key": self.key.to_record(),
            "execution": self.execution.to_record(),
            "test_pass_receipt": (
                self.test_pass_receipt.to_record() if self.test_pass_receipt else None
            ),
            "test_execution_key": (
                self.test_execution_key.to_record() if self.test_execution_key else None
            ),
            "status": self.status,
            "artifact_cids": self.artifact_cids,
            "reason_codes": self.reason_codes,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TestReceipt:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="test receipt",
        )
        _reject_unknown(
            payload,
            {
                "key",
                "execution",
                "test_pass_receipt",
                "test_execution_key",
                "status",
                "artifact_cids",
                "reason_codes",
                "receipt_id",
            },
            artifact_name="test receipt",
        )
        execution = payload.get("execution")
        source = payload.get("test_pass_receipt")
        source_key = payload.get("test_execution_key")
        result = cls(
            key=_key(payload.get("key")),
            execution=_observation(execution),
            test_pass_receipt=_test_pass_receipt(source) if source is not None else None,
            test_execution_key=(
                _test_execution_key(source_key) if source_key is not None else None
            ),
            artifact_cids=tuple(payload.get("artifact_cids") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        _check_projection(payload, field_name="status", actual=result.status)
        _check_identity(
            payload,
            result.receipt_id,
            names=("receipt_id", "content_id"),
            artifact_name="test receipt",
        )
        return result


def _formal_proof_receipt(value: Any) -> FormalProofReceipt:
    if isinstance(value, FormalProofReceipt):
        return value
    if isinstance(value, Mapping):
        return FormalProofReceipt.from_dict(value)
    raise VerificationContractError(
        "formal_proof_receipt must be an existing canonical ProofReceipt"
    )


@dataclass(frozen=True)
class ProofReceipt(_VerificationContract):
    """Verification wrapper over the existing authoritative proof contract.

    A provider's declared verdict never becomes ``proved``.  That projection
    requires current accepted evidence in the existing assurance lattice.
    """

    SCHEMA: ClassVar[str] = PROOF_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = PROOF_RECEIPT_INTERFACE

    key: VerificationReceiptKey
    execution: DirectExecutionObservation
    formal_proof_receipt: FormalProofReceipt | None = None
    artifact_cids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _key(self.key))
        _validate_receipt_kind(self.key, VerificationReceiptKind.PROOF)
        execution = _observation(self.execution)
        object.__setattr__(self, "execution", execution)
        _validate_execution_binding(self.key, execution)
        if self.formal_proof_receipt is None:
            _direct_check_status(execution.terminal_status, proof=True)
        else:
            formal = _formal_proof_receipt(self.formal_proof_receipt)
            object.__setattr__(self, "formal_proof_receipt", formal)
            if formal.repository_tree_id != self.key.repository_tree_cid:
                raise VerificationIdentityError(
                    "formal proof receipt uses a different repository tree"
                )
            if formal.obligation_id != self.key.proof_obligation_cid:
                raise VerificationIdentityError(
                    "formal proof receipt uses a different proof obligation"
                )
        object.__setattr__(self, "artifact_cids", _receipt_artifacts(self.artifact_cids))
        object.__setattr__(
            self, "reason_codes", _reason_codes(self.reason_codes, field_name="reason_codes")
        )
        _bounded(self, artifact_name="proof receipt")

    @property
    def status(self) -> TerminalStatus:
        if self.formal_proof_receipt is None:
            return self.execution.terminal_status
        receipt = self.formal_proof_receipt
        if receipt.freshness is not EvidenceFreshness.CURRENT:
            return TerminalStatus.STALE
        if (
            receipt.verdict is ProofVerdict.PROVED
            and receipt.authoritative_assurance.satisfies(
                AssuranceLevel.SOLVER_CHECKED
            )
        ):
            return TerminalStatus.PROVED
        if receipt.authoritative_verdict is ProofVerdict.DISPROVED:
            return TerminalStatus.DISPROVED
        if any(item.simulated for item in receipt.evidence):
            return TerminalStatus.SIMULATED
        if receipt.verdict is ProofVerdict.CANCELLED:
            return TerminalStatus.CANCELLED
        if receipt.verdict is ProofVerdict.UNSUPPORTED:
            return TerminalStatus.UNAVAILABLE
        if receipt.verdict is ProofVerdict.ERROR:
            return TerminalStatus.INVALID
        return TerminalStatus.UNKNOWN

    @property
    def terminal_success(self) -> bool:
        return self.status is TerminalStatus.PROVED

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "key": self.key.to_record(),
            "execution": self.execution.to_record(),
            "formal_proof_receipt": (
                self.formal_proof_receipt.to_record()
                if self.formal_proof_receipt
                else None
            ),
            "status": self.status,
            "artifact_cids": self.artifact_cids,
            "reason_codes": self.reason_codes,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProofReceipt:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="proof receipt",
        )
        _reject_unknown(
            payload,
            {
                "key",
                "execution",
                "formal_proof_receipt",
                "status",
                "artifact_cids",
                "reason_codes",
                "receipt_id",
            },
            artifact_name="proof receipt",
        )
        execution = payload.get("execution")
        source = payload.get("formal_proof_receipt")
        result = cls(
            key=_key(payload.get("key")),
            execution=_observation(execution),
            formal_proof_receipt=(
                _formal_proof_receipt(source) if source is not None else None
            ),
            artifact_cids=tuple(payload.get("artifact_cids") or ()),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        _check_projection(payload, field_name="status", actual=result.status)
        _check_identity(
            payload,
            result.receipt_id,
            names=("receipt_id", "content_id"),
            artifact_name="proof receipt",
        )
        return result


def _diagnostic_value(value: Any, *, field_name: str) -> Mapping[str, Any]:
    result = _mapping(value, field_name=field_name, required=True)
    if set(result).difference({"state", "value"}):
        raise VerificationContractError(
            f"{field_name} contains unsupported diagnostic fields"
        )
    state = _enum(
        result.get("state"), DiagnosticValueState, field_name=f"{field_name}.state"
    )
    has_value = "value" in result
    if state is DiagnosticValueState.PRESENT and not has_value:
        raise VerificationContractError(f"{field_name} present state requires value")
    if state is not DiagnosticValueState.PRESENT and has_value:
        raise VerificationContractError(
            f"{field_name} non-present states cannot embed a value"
        )
    normalized = {"state": state.value}
    if has_value:
        normalized["value"] = result["value"]
    return MappingProxyType(normalized)


def _source_spans(values: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise VerificationContractError("source_spans must be a sequence")
    if len(values) > 128:
        raise VerificationBoundsError("source_spans exceeds item bound")
    spans: list[Mapping[str, Any]] = []
    for index, value in enumerate(values):
        span = _mapping(value, field_name=f"source_spans[{index}]", required=True)
        allowed = {"path", "start_line", "end_line", "artifact_cid", "symbol"}
        if set(span).difference(allowed):
            raise VerificationContractError("source span contains unsupported fields")
        path = _text(span.get("path", ""), field_name="source span path", maximum=1_024)
        if path.startswith(("/", "\\")) or ".." in path.replace("\\", "/").split("/"):
            raise VerificationContractError("source span path must be repository-relative")
        start = _integer(span.get("start_line"), field_name="source span start", minimum=1)
        end = _integer(span.get("end_line"), field_name="source span end", minimum=1)
        if end < start:
            raise VerificationContractError("source span end precedes start")
        normalized = MappingProxyType(
            {
                "path": path,
                "start_line": start,
                "end_line": end,
                "artifact_cid": _cid(
                    span.get("artifact_cid", ""), field_name="source span artifact_cid"
                ),
                "symbol": _text(
                    span.get("symbol", ""),
                    field_name="source span symbol",
                    required=False,
                    maximum=512,
                ),
            }
        )
        spans.append(normalized)
    identities = [canonical_json_bytes(span) for span in spans]
    if len(identities) != len(set(identities)):
        raise VerificationContractError("source_spans contains duplicates")
    return tuple(sorted(spans, key=canonical_json_bytes))


@dataclass(frozen=True)
class CounterexampleReceipt(_VerificationContract):
    """Compact failure reproducer; complete logs remain artifact-addressed."""

    SCHEMA: ClassVar[str] = COUNTEREXAMPLE_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = COUNTEREXAMPLE_RECEIPT_INTERFACE

    failed_key_cid: str
    failed_receipt_cid: str
    failed_selector: str
    failure_identity_cid: str
    relevant_symbol_version_cids: tuple[str, ...]
    minimized_traceback: tuple[str, ...]
    relevant_assertion: str
    relevant_input: Mapping[str, Any]
    expected_output: Mapping[str, Any]
    observed_output: Mapping[str, Any]
    source_spans: tuple[Mapping[str, Any], ...]
    environment_cid: str
    dependency_lock_cid: str
    reproduction_argv: tuple[str, ...]
    artifact_cids: tuple[str, ...]
    minimized: bool
    failed_obligation_cid: str = ""
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "failed_key_cid",
            "failed_receipt_cid",
            "failure_identity_cid",
            "environment_cid",
            "dependency_lock_cid",
        ):
            object.__setattr__(self, name, _cid(getattr(self, name), field_name=name))
        object.__setattr__(
            self,
            "failed_obligation_cid",
            _cid(
                self.failed_obligation_cid,
                field_name="failed_obligation_cid",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "failed_selector",
            _text(self.failed_selector, field_name="failed_selector", maximum=2_048),
        )
        object.__setattr__(
            self,
            "relevant_symbol_version_cids",
            _cids(
                self.relevant_symbol_version_cids,
                field_name="relevant_symbol_version_cids",
            ),
        )
        object.__setattr__(
            self,
            "minimized_traceback",
            _strings(
                self.minimized_traceback,
                field_name="minimized_traceback",
                required=True,
                preserve_order=True,
                maximum=64,
                item_bytes=2_048,
            ),
        )
        object.__setattr__(
            self,
            "relevant_assertion",
            _text(
                self.relevant_assertion,
                field_name="relevant_assertion",
                maximum=4_096,
            ),
        )
        for name in ("relevant_input", "expected_output", "observed_output"):
            object.__setattr__(
                self,
                name,
                _diagnostic_value(getattr(self, name), field_name=name),
            )
        object.__setattr__(self, "source_spans", _source_spans(self.source_spans))
        object.__setattr__(
            self,
            "reproduction_argv",
            _strings(
                self.reproduction_argv,
                field_name="reproduction_argv",
                required=True,
                preserve_order=True,
            ),
        )
        object.__setattr__(self, "artifact_cids", _receipt_artifacts(self.artifact_cids))
        object.__setattr__(self, "minimized", _boolean(self.minimized, field_name="minimized"))
        object.__setattr__(
            self, "reason_codes", _reason_codes(self.reason_codes, field_name="reason_codes")
        )
        _bounded(
            self,
            artifact_name="counterexample receipt",
            maximum=MAX_COUNTEREXAMPLE_BYTES,
        )

    @property
    def counterexample_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "failed_key_cid": self.failed_key_cid,
            "failed_receipt_cid": self.failed_receipt_cid,
            "failed_selector": self.failed_selector,
            "failed_obligation_cid": self.failed_obligation_cid,
            "failure_identity_cid": self.failure_identity_cid,
            "relevant_symbol_version_cids": self.relevant_symbol_version_cids,
            "minimized_traceback": self.minimized_traceback,
            "relevant_assertion": self.relevant_assertion,
            "relevant_input": self.relevant_input,
            "expected_output": self.expected_output,
            "observed_output": self.observed_output,
            "source_spans": self.source_spans,
            "environment_cid": self.environment_cid,
            "dependency_lock_cid": self.dependency_lock_cid,
            "reproduction_argv": self.reproduction_argv,
            "artifact_cids": self.artifact_cids,
            "minimized": self.minimized,
            "reason_codes": self.reason_codes,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "counterexample_id": self.counterexample_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CounterexampleReceipt:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="counterexample receipt",
        )
        fields = {
            "failed_key_cid",
            "failed_receipt_cid",
            "failed_selector",
            "failed_obligation_cid",
            "failure_identity_cid",
            "relevant_symbol_version_cids",
            "minimized_traceback",
            "relevant_assertion",
            "relevant_input",
            "expected_output",
            "observed_output",
            "source_spans",
            "environment_cid",
            "dependency_lock_cid",
            "reproduction_argv",
            "artifact_cids",
            "minimized",
            "reason_codes",
            "counterexample_id",
        }
        _reject_unknown(payload, fields, artifact_name="counterexample receipt")
        result = cls(
            failed_key_cid=payload.get("failed_key_cid", ""),
            failed_receipt_cid=payload.get("failed_receipt_cid", ""),
            failed_selector=payload.get("failed_selector", ""),
            failed_obligation_cid=payload.get("failed_obligation_cid", ""),
            failure_identity_cid=payload.get("failure_identity_cid", ""),
            relevant_symbol_version_cids=tuple(
                payload.get("relevant_symbol_version_cids") or ()
            ),
            minimized_traceback=tuple(payload.get("minimized_traceback") or ()),
            relevant_assertion=payload.get("relevant_assertion", ""),
            relevant_input=payload.get("relevant_input") or {},
            expected_output=payload.get("expected_output") or {},
            observed_output=payload.get("observed_output") or {},
            source_spans=tuple(payload.get("source_spans") or ()),
            environment_cid=payload.get("environment_cid", ""),
            dependency_lock_cid=payload.get("dependency_lock_cid", ""),
            reproduction_argv=tuple(payload.get("reproduction_argv") or ()),
            artifact_cids=tuple(payload.get("artifact_cids") or ()),
            minimized=payload.get("minimized"),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        _check_identity(
            payload,
            result.counterexample_id,
            names=("counterexample_id", "content_id"),
            artifact_name="counterexample receipt",
        )
        return result


@dataclass(frozen=True)
class CacheReuseDecision(_VerificationContract):
    """Explicit exact-key cache disposition; absence never becomes reuse."""

    SCHEMA: ClassVar[str] = CACHE_REUSE_DECISION_SCHEMA
    INTERFACE: ClassVar[str] = CACHE_REUSE_DECISION_INTERFACE

    key_cid: str
    disposition: CacheReuseDisposition
    reason_codes: tuple[str, ...]
    receipt_cid: str = ""
    candidate_status: TerminalStatus | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "key_cid", _cid(self.key_cid, field_name="key_cid"))
        object.__setattr__(
            self,
            "disposition",
            _enum(
                self.disposition,
                CacheReuseDisposition,
                field_name="disposition",
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _reason_codes(self.reason_codes, field_name="reason_codes", required=True),
        )
        object.__setattr__(
            self,
            "receipt_cid",
            _cid(self.receipt_cid, field_name="receipt_cid", required=False),
        )
        if self.candidate_status is not None:
            object.__setattr__(
                self,
                "candidate_status",
                _enum(
                    self.candidate_status,
                    TerminalStatus,
                    field_name="candidate_status",
                ),
            )
        if self.disposition is CacheReuseDisposition.REUSED:
            if not self.receipt_cid:
                raise VerificationContractError("reused decision requires receipt_cid")
            if self.candidate_status not in {
                TerminalStatus.PASSED,
                TerminalStatus.PROVED,
            }:
                raise VerificationContractError(
                    "reused decision requires a successful terminal candidate"
                )
        if self.disposition is CacheReuseDisposition.MISSING and self.receipt_cid:
            raise VerificationContractError("missing decision cannot name a receipt")
        if (
            self.disposition is CacheReuseDisposition.SIMULATED
            and self.candidate_status is not TerminalStatus.SIMULATED
        ):
            raise VerificationContractError(
                "simulated disposition requires simulated candidate status"
            )
        _bounded(self, artifact_name="cache reuse decision")

    @property
    def reusable(self) -> bool:
        return self.disposition is CacheReuseDisposition.REUSED

    @property
    def decision_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "key_cid": self.key_cid,
            "disposition": self.disposition,
            "reason_codes": self.reason_codes,
            "receipt_cid": self.receipt_cid,
            "candidate_status": self.candidate_status,
            "reusable": self.reusable,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "decision_id": self.decision_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CacheReuseDecision:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="cache reuse decision",
        )
        _reject_unknown(
            payload,
            {
                "key_cid",
                "disposition",
                "reason_codes",
                "receipt_cid",
                "candidate_status",
                "reusable",
                "decision_id",
            },
            artifact_name="cache reuse decision",
        )
        result = cls(
            key_cid=payload.get("key_cid", ""),
            disposition=payload.get("disposition", ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
            receipt_cid=payload.get("receipt_cid", ""),
            candidate_status=payload.get("candidate_status"),
        )
        if "reusable" in payload and not isinstance(payload["reusable"], bool):
            raise VerificationContractError("reusable projection must be a boolean")
        _check_projection(payload, field_name="reusable", actual=result.reusable)
        _check_identity(
            payload,
            result.decision_id,
            names=("decision_id", "content_id"),
            artifact_name="cache reuse decision",
        )
        return result


@dataclass(frozen=True)
class ModelRouteDecision(_VerificationContract):
    """Provider-neutral capability route for the next repair."""

    SCHEMA: ClassVar[str] = MODEL_ROUTE_DECISION_SCHEMA
    INTERFACE: ClassVar[str] = MODEL_ROUTE_DECISION_INTERFACE

    route: ModelRoute
    considered_routes: tuple[ModelRoute, ...]
    decisive_reason_codes: tuple[str, ...]
    required_capabilities: tuple[str, ...]
    context_token_estimate: int
    policy_cid: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "route", _enum(self.route, ModelRoute, field_name="route")
        )
        if isinstance(self.considered_routes, (str, bytes)) or not isinstance(
            self.considered_routes, Sequence
        ):
            raise VerificationContractError("considered_routes must be a sequence")
        routes = tuple(
            _enum(item, ModelRoute, field_name="considered_routes")
            for item in self.considered_routes
        )
        if not routes or len(routes) != len(set(routes)):
            raise VerificationContractError(
                "considered_routes must be nonempty and unique"
            )
        if self.route not in routes:
            raise VerificationContractError("selected route was not considered")
        object.__setattr__(self, "considered_routes", routes)
        object.__setattr__(
            self,
            "decisive_reason_codes",
            _reason_codes(
                self.decisive_reason_codes,
                field_name="decisive_reason_codes",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "required_capabilities",
            _strings(
                self.required_capabilities,
                field_name="required_capabilities",
                maximum=128,
                item_bytes=256,
            ),
        )
        object.__setattr__(
            self,
            "context_token_estimate",
            _integer(
                self.context_token_estimate,
                field_name="context_token_estimate",
            ),
        )
        object.__setattr__(
            self, "policy_cid", _cid(self.policy_cid, field_name="policy_cid")
        )
        _bounded(self, artifact_name="model route decision")

    @property
    def requires_human_review(self) -> bool:
        return self.route is ModelRoute.HUMAN_REVIEW_REQUIRED

    @property
    def decision_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "route": self.route,
            "considered_routes": self.considered_routes,
            "decisive_reason_codes": self.decisive_reason_codes,
            "required_capabilities": self.required_capabilities,
            "context_token_estimate": self.context_token_estimate,
            "policy_cid": self.policy_cid,
            "requires_human_review": self.requires_human_review,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "decision_id": self.decision_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelRouteDecision:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="model route decision",
        )
        _reject_unknown(
            payload,
            {
                "route",
                "considered_routes",
                "decisive_reason_codes",
                "required_capabilities",
                "context_token_estimate",
                "policy_cid",
                "requires_human_review",
                "decision_id",
            },
            artifact_name="model route decision",
        )
        result = cls(
            route=payload.get("route", ""),
            considered_routes=tuple(payload.get("considered_routes") or ()),
            decisive_reason_codes=tuple(payload.get("decisive_reason_codes") or ()),
            required_capabilities=tuple(payload.get("required_capabilities") or ()),
            context_token_estimate=payload.get("context_token_estimate", -1),
            policy_cid=payload.get("policy_cid", ""),
        )
        if "requires_human_review" in payload and not isinstance(
            payload["requires_human_review"], bool
        ):
            raise VerificationContractError(
                "requires_human_review projection must be a boolean"
            )
        _check_projection(
            payload,
            field_name="requires_human_review",
            actual=result.requires_human_review,
        )
        _check_identity(
            payload,
            result.decision_id,
            names=("decision_id", "content_id"),
            artifact_name="model route decision",
        )
        return result


def _receipt_keys(values: Any) -> tuple[VerificationReceiptKey, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise VerificationContractError("required_receipt_keys must be a sequence")
    if not values or len(values) > MAX_COLLECTION_ITEMS:
        raise VerificationBoundsError(
            "required_receipt_keys must be nonempty and within its item bound"
        )
    keys = tuple(_key(item, field_name="required_receipt_keys") for item in values)
    ids = tuple(item.key_id for item in keys)
    if len(ids) != len(set(ids)):
        raise VerificationContractError("required_receipt_keys contains duplicates")
    return tuple(sorted(keys, key=lambda item: item.key_id))


def _reuse_decisions(values: Any) -> tuple[CacheReuseDecision, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise VerificationContractError("cache_reuse_decisions must be a sequence")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise VerificationBoundsError("cache_reuse_decisions exceeds item bound")
    result: list[CacheReuseDecision] = []
    for item in values:
        if isinstance(item, CacheReuseDecision):
            decision = item
        elif isinstance(item, Mapping):
            decision = CacheReuseDecision.from_dict(item)
        else:
            raise VerificationContractError(
                "cache_reuse_decisions contains an invalid record"
            )
        result.append(decision)
    ids = tuple(item.decision_id for item in result)
    if len(ids) != len(set(ids)):
        raise VerificationContractError("cache_reuse_decisions contains duplicates")
    return tuple(sorted(result, key=lambda item: (item.key_cid, item.decision_id)))


def _timeout_mapping(value: Any, *, field_name: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise VerificationContractError(f"{field_name} must be a mapping")
    if len(value) > MAX_COLLECTION_ITEMS:
        raise VerificationBoundsError(f"{field_name} exceeds item bound")
    result: dict[str, int] = {}
    for raw_key in sorted(value):
        key = _text(raw_key, field_name=f"{field_name} key", maximum=512)
        result[key] = _integer(
            value[raw_key],
            field_name=f"{field_name}.{key}",
            minimum=1,
            maximum=MAX_DURATION_MS,
        )
    return MappingProxyType(result)


def _dependency_dag(value: Any) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise VerificationContractError("dependency_dag must be a mapping")
    if len(value) > MAX_COLLECTION_ITEMS:
        raise VerificationBoundsError("dependency_dag exceeds item bound")
    result: dict[str, tuple[str, ...]] = {}
    for raw_step in sorted(value):
        step = _text(raw_step, field_name="dependency_dag step", maximum=512)
        dependencies = _strings(
            value[raw_step],
            field_name=f"dependency_dag.{step}",
            maximum=MAX_COLLECTION_ITEMS,
            item_bytes=512,
        )
        if step in dependencies:
            raise VerificationContractError("dependency_dag contains a self edge")
        result[step] = dependencies
    step_ids = set(result)
    for dependencies in result.values():
        if not set(dependencies).issubset(step_ids):
            raise VerificationContractError(
                "dependency_dag references an undeclared step"
            )
    # Exercise deterministic Kahn traversal to reject cycles now.
    _topological_order(result)
    return MappingProxyType(result)


def _topological_order(dag: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    pending = {step: set(dependencies) for step, dependencies in dag.items()}
    order: list[str] = []
    while pending:
        ready = sorted(step for step, dependencies in pending.items() if not dependencies)
        if not ready:
            raise VerificationContractError("dependency_dag contains a cycle")
        for step in ready:
            order.append(step)
            pending.pop(step)
        for dependencies in pending.values():
            dependencies.difference_update(ready)
    return tuple(order)


@dataclass(frozen=True)
class VerificationPlan(_VerificationContract):
    """Deterministic, side-effect-free incremental verification plan."""

    SCHEMA: ClassVar[str] = VERIFICATION_PLAN_SCHEMA
    INTERFACE: ClassVar[str] = VERIFICATION_PLAN_INTERFACE

    repository_tree_cid: str
    semantic_state_root_cid: str
    environment_cid: str
    dependency_lock_cid: str
    required_receipt_keys: tuple[VerificationReceiptKey, ...]
    cache_reuse_decisions: tuple[CacheReuseDecision, ...]
    affected_tests: tuple[str, ...]
    fallback_tests: tuple[str, ...]
    required_static_checks: tuple[str, ...]
    required_type_checks: tuple[str, ...]
    affected_proof_obligation_cids: tuple[str, ...]
    full_suite_required: bool
    full_suite_reason_codes: tuple[str, ...]
    human_review_required: bool
    human_review_reason_codes: tuple[str, ...]
    expected_cpu_millis: int
    expected_memory_bytes: int
    expected_processes: int
    expected_proof_slots: int
    expected_artifact_bytes: int
    step_timeouts_ms: Mapping[str, int]
    max_execution_time_ms: int
    dependency_dag: Mapping[str, tuple[str, ...]]
    acceptance_criteria: tuple[str, ...]
    policy_cid: str

    def __post_init__(self) -> None:
        for name in (
            "repository_tree_cid",
            "semantic_state_root_cid",
            "environment_cid",
            "dependency_lock_cid",
            "policy_cid",
        ):
            object.__setattr__(self, name, _cid(getattr(self, name), field_name=name))
        object.__setattr__(
            self, "required_receipt_keys", _receipt_keys(self.required_receipt_keys)
        )
        for key in self.required_receipt_keys:
            if (
                key.repository_tree_cid != self.repository_tree_cid
                or key.semantic_state_root_cid != self.semantic_state_root_cid
                or key.environment_cid != self.environment_cid
                or key.dependency_lock_cid != self.dependency_lock_cid
            ):
                raise VerificationIdentityError(
                    "required receipt key does not match plan identities"
                )
        object.__setattr__(
            self,
            "cache_reuse_decisions",
            _reuse_decisions(self.cache_reuse_decisions),
        )
        for name in (
            "affected_tests",
            "fallback_tests",
            "required_static_checks",
            "required_type_checks",
        ):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), field_name=name, item_bytes=2_048),
            )
        object.__setattr__(
            self,
            "affected_proof_obligation_cids",
            _cids(
                self.affected_proof_obligation_cids,
                field_name="affected_proof_obligation_cids",
            ),
        )
        object.__setattr__(
            self,
            "full_suite_required",
            _boolean(self.full_suite_required, field_name="full_suite_required"),
        )
        object.__setattr__(
            self,
            "full_suite_reason_codes",
            _reason_codes(
                self.full_suite_reason_codes,
                field_name="full_suite_reason_codes",
                required=self.full_suite_required,
            ),
        )
        object.__setattr__(
            self,
            "human_review_required",
            _boolean(self.human_review_required, field_name="human_review_required"),
        )
        object.__setattr__(
            self,
            "human_review_reason_codes",
            _reason_codes(
                self.human_review_reason_codes,
                field_name="human_review_reason_codes",
                required=self.human_review_required,
            ),
        )
        for name, minimum in (
            ("expected_cpu_millis", 0),
            ("expected_memory_bytes", 0),
            ("expected_processes", 1),
            ("expected_proof_slots", 0),
            ("expected_artifact_bytes", 0),
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), field_name=name, minimum=minimum),
            )
        object.__setattr__(
            self,
            "max_execution_time_ms",
            _integer(
                self.max_execution_time_ms,
                field_name="max_execution_time_ms",
                minimum=1,
                maximum=MAX_DURATION_MS,
            ),
        )
        object.__setattr__(
            self,
            "step_timeouts_ms",
            _timeout_mapping(self.step_timeouts_ms, field_name="step_timeouts_ms"),
        )
        object.__setattr__(self, "dependency_dag", _dependency_dag(self.dependency_dag))
        if set(self.step_timeouts_ms) != set(self.dependency_dag):
            raise VerificationContractError(
                "step_timeouts_ms must cover the dependency DAG exactly"
            )
        if any(
            timeout > self.max_execution_time_ms
            for timeout in self.step_timeouts_ms.values()
        ):
            raise VerificationBoundsError(
                "step timeout exceeds maximum execution time"
            )
        object.__setattr__(
            self,
            "acceptance_criteria",
            _strings(
                self.acceptance_criteria,
                field_name="acceptance_criteria",
                required=True,
                preserve_order=True,
                maximum=128,
                item_bytes=1_024,
            ),
        )
        _bounded(self, artifact_name="verification plan")

    @property
    def execution_order(self) -> tuple[str, ...]:
        return _topological_order(self.dependency_dag)

    @property
    def plan_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "repository_tree_cid": self.repository_tree_cid,
            "semantic_state_root_cid": self.semantic_state_root_cid,
            "environment_cid": self.environment_cid,
            "dependency_lock_cid": self.dependency_lock_cid,
            "required_receipt_keys": tuple(
                item.to_record() for item in self.required_receipt_keys
            ),
            "cache_reuse_decisions": tuple(
                item.to_record() for item in self.cache_reuse_decisions
            ),
            "affected_tests": self.affected_tests,
            "fallback_tests": self.fallback_tests,
            "required_static_checks": self.required_static_checks,
            "required_type_checks": self.required_type_checks,
            "affected_proof_obligation_cids": self.affected_proof_obligation_cids,
            "full_suite_required": self.full_suite_required,
            "full_suite_reason_codes": self.full_suite_reason_codes,
            "human_review_required": self.human_review_required,
            "human_review_reason_codes": self.human_review_reason_codes,
            "expected_cpu_millis": self.expected_cpu_millis,
            "expected_memory_bytes": self.expected_memory_bytes,
            "expected_processes": self.expected_processes,
            "expected_proof_slots": self.expected_proof_slots,
            "expected_artifact_bytes": self.expected_artifact_bytes,
            "step_timeouts_ms": self.step_timeouts_ms,
            "max_execution_time_ms": self.max_execution_time_ms,
            "dependency_dag": self.dependency_dag,
            "execution_order": self.execution_order,
            "acceptance_criteria": self.acceptance_criteria,
            "policy_cid": self.policy_cid,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "plan_id": self.plan_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationPlan:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="verification plan",
        )
        fields = {
            "repository_tree_cid",
            "semantic_state_root_cid",
            "environment_cid",
            "dependency_lock_cid",
            "required_receipt_keys",
            "cache_reuse_decisions",
            "affected_tests",
            "fallback_tests",
            "required_static_checks",
            "required_type_checks",
            "affected_proof_obligation_cids",
            "full_suite_required",
            "full_suite_reason_codes",
            "human_review_required",
            "human_review_reason_codes",
            "expected_cpu_millis",
            "expected_memory_bytes",
            "expected_processes",
            "expected_proof_slots",
            "expected_artifact_bytes",
            "step_timeouts_ms",
            "max_execution_time_ms",
            "dependency_dag",
            "execution_order",
            "acceptance_criteria",
            "policy_cid",
            "plan_id",
        }
        _reject_unknown(payload, fields, artifact_name="verification plan")
        result = cls(
            repository_tree_cid=payload.get("repository_tree_cid", ""),
            semantic_state_root_cid=payload.get("semantic_state_root_cid", ""),
            environment_cid=payload.get("environment_cid", ""),
            dependency_lock_cid=payload.get("dependency_lock_cid", ""),
            required_receipt_keys=tuple(payload.get("required_receipt_keys") or ()),
            cache_reuse_decisions=tuple(
                payload.get("cache_reuse_decisions") or ()
            ),
            affected_tests=tuple(payload.get("affected_tests") or ()),
            fallback_tests=tuple(payload.get("fallback_tests") or ()),
            required_static_checks=tuple(payload.get("required_static_checks") or ()),
            required_type_checks=tuple(payload.get("required_type_checks") or ()),
            affected_proof_obligation_cids=tuple(
                payload.get("affected_proof_obligation_cids") or ()
            ),
            full_suite_required=payload.get("full_suite_required"),
            full_suite_reason_codes=tuple(
                payload.get("full_suite_reason_codes") or ()
            ),
            human_review_required=payload.get("human_review_required"),
            human_review_reason_codes=tuple(
                payload.get("human_review_reason_codes") or ()
            ),
            expected_cpu_millis=payload.get("expected_cpu_millis", -1),
            expected_memory_bytes=payload.get("expected_memory_bytes", -1),
            expected_processes=payload.get("expected_processes", 0),
            expected_proof_slots=payload.get("expected_proof_slots", -1),
            expected_artifact_bytes=payload.get("expected_artifact_bytes", -1),
            step_timeouts_ms=payload.get("step_timeouts_ms") or {},
            max_execution_time_ms=payload.get("max_execution_time_ms", 0),
            dependency_dag=payload.get("dependency_dag") or {},
            acceptance_criteria=tuple(payload.get("acceptance_criteria") or ()),
            policy_cid=payload.get("policy_cid", ""),
        )
        _check_projection(payload, field_name="execution_order", actual=result.execution_order)
        _check_identity(
            payload,
            result.plan_id,
            names=("plan_id", "content_id"),
            artifact_name="verification plan",
        )
        return result


VerificationReceipt: TypeAlias = (
    StaticAnalysisReceipt | TypeCheckReceipt | TestReceipt | ProofReceipt
)

_RECEIPT_TYPES_BY_SCHEMA: Final[Mapping[str, type[_VerificationContract]]] = MappingProxyType(
    {
        STATIC_ANALYSIS_RECEIPT_SCHEMA: StaticAnalysisReceipt,
        TYPE_CHECK_RECEIPT_SCHEMA: TypeCheckReceipt,
        TEST_RECEIPT_SCHEMA: TestReceipt,
        PROOF_RECEIPT_SCHEMA: ProofReceipt,
    }
)


def _verification_receipt(value: Any) -> VerificationReceipt:
    if isinstance(
        value, (StaticAnalysisReceipt, TypeCheckReceipt, TestReceipt, ProofReceipt)
    ):
        return value
    if not isinstance(value, Mapping):
        raise VerificationContractError("receipt must be a canonical receipt record")
    receipt_type = _RECEIPT_TYPES_BY_SCHEMA.get(value.get("schema"))
    if receipt_type is None:
        raise VerificationContractError("receipt has an unsupported schema")
    result = receipt_type.from_dict(value)  # type: ignore[attr-defined]
    assert isinstance(
        result, (StaticAnalysisReceipt, TypeCheckReceipt, TestReceipt, ProofReceipt)
    )
    return result


def _verification_receipts(values: Any) -> tuple[VerificationReceipt, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise VerificationContractError("receipts must be a sequence")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise VerificationBoundsError("receipts exceeds item bound")
    receipts = tuple(_verification_receipt(item) for item in values)
    ids = tuple(item.receipt_id for item in receipts)
    if len(ids) != len(set(ids)):
        raise VerificationContractError("receipts contains duplicate identities")
    key_ids = tuple(item.key.key_id for item in receipts)
    if len(key_ids) != len(set(key_ids)):
        raise VerificationContractError("receipts contains more than one result per key")
    return tuple(sorted(receipts, key=lambda item: (item.key.key_id, item.receipt_id)))


def _counterexamples(values: Any) -> tuple[CounterexampleReceipt, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise VerificationContractError("counterexamples must be a sequence")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise VerificationBoundsError("counterexamples exceeds item bound")
    result: list[CounterexampleReceipt] = []
    for item in values:
        if isinstance(item, CounterexampleReceipt):
            counterexample = item
        elif isinstance(item, Mapping):
            counterexample = CounterexampleReceipt.from_dict(item)
        else:
            raise VerificationContractError(
                "counterexamples contains an invalid record"
            )
        result.append(counterexample)
    ids = tuple(item.counterexample_id for item in result)
    if len(ids) != len(set(ids)):
        raise VerificationContractError("counterexamples contains duplicates")
    return tuple(sorted(result, key=lambda item: item.counterexample_id))


@dataclass(frozen=True)
class VerificationBundle(_VerificationContract):
    """Exact required receipts plus explicit unresolved requirements."""

    SCHEMA: ClassVar[str] = VERIFICATION_BUNDLE_SCHEMA
    INTERFACE: ClassVar[str] = VERIFICATION_BUNDLE_INTERFACE

    plan_cid: str
    repository_tree_cid: str
    environment_cid: str
    required_check_key_cids: tuple[str, ...]
    receipts: tuple[VerificationReceipt, ...]
    reused_receipt_cids: tuple[str, ...]
    executed_receipt_cids: tuple[str, ...]
    counterexamples: tuple[CounterexampleReceipt, ...]
    unresolved_requirement_ids: tuple[str, ...]
    mandatory_fallback_pending: bool
    human_review_required: bool
    policy_cid: str

    def __post_init__(self) -> None:
        for name in ("plan_cid", "repository_tree_cid", "environment_cid", "policy_cid"):
            object.__setattr__(self, name, _cid(getattr(self, name), field_name=name))
        object.__setattr__(
            self,
            "required_check_key_cids",
            _cids(
                self.required_check_key_cids,
                field_name="required_check_key_cids",
                required=True,
            ),
        )
        object.__setattr__(self, "receipts", _verification_receipts(self.receipts))
        required = set(self.required_check_key_cids)
        receipt_ids = {item.receipt_id for item in self.receipts}
        receipt_key_ids = {item.key.key_id for item in self.receipts}
        if not receipt_key_ids.issubset(required):
            raise VerificationIdentityError(
                "bundle contains a receipt outside the required check set"
            )
        for receipt in self.receipts:
            if (
                receipt.key.repository_tree_cid != self.repository_tree_cid
                or receipt.key.environment_cid != self.environment_cid
            ):
                raise VerificationIdentityError(
                    "bundle contains mixed repository tree or environment receipts"
                )
        for name in ("reused_receipt_cids", "executed_receipt_cids"):
            object.__setattr__(
                self,
                name,
                _cids(getattr(self, name), field_name=name),
            )
            if not set(getattr(self, name)).issubset(receipt_ids):
                raise VerificationIdentityError(
                    f"{name} names a receipt not carried by the bundle"
                )
        if set(self.reused_receipt_cids) & set(self.executed_receipt_cids):
            raise VerificationContractError(
                "reused and executed receipt sets must be disjoint"
            )
        if set(self.reused_receipt_cids) | set(self.executed_receipt_cids) != receipt_ids:
            raise VerificationContractError(
                "every bundled receipt must be classified as reused or executed"
            )
        object.__setattr__(
            self, "counterexamples", _counterexamples(self.counterexamples)
        )
        if not {
            item.failed_receipt_cid for item in self.counterexamples
        }.issubset(receipt_ids):
            raise VerificationIdentityError(
                "counterexample references a receipt outside the bundle"
            )
        object.__setattr__(
            self,
            "unresolved_requirement_ids",
            _strings(
                self.unresolved_requirement_ids,
                field_name="unresolved_requirement_ids",
                item_bytes=512,
            ),
        )
        missing_keys = required - receipt_key_ids
        if not missing_keys.issubset(set(self.unresolved_requirement_ids)):
            raise VerificationContractError(
                "missing required receipt keys must be explicit unresolved requirements"
            )
        object.__setattr__(
            self,
            "mandatory_fallback_pending",
            _boolean(
                self.mandatory_fallback_pending,
                field_name="mandatory_fallback_pending",
            ),
        )
        object.__setattr__(
            self,
            "human_review_required",
            _boolean(self.human_review_required, field_name="human_review_required"),
        )
        _bounded(self, artifact_name="verification bundle")

    @property
    def structurally_complete(self) -> bool:
        return bool(
            len(self.receipts) == len(self.required_check_key_cids)
            and not self.unresolved_requirement_ids
            and not self.mandatory_fallback_pending
            and not self.human_review_required
            and all(item.terminal_success for item in self.receipts)
        )

    @property
    def bundle_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "plan_cid": self.plan_cid,
            "repository_tree_cid": self.repository_tree_cid,
            "environment_cid": self.environment_cid,
            "required_check_key_cids": self.required_check_key_cids,
            "receipts": tuple(item.to_record() for item in self.receipts),
            "reused_receipt_cids": self.reused_receipt_cids,
            "executed_receipt_cids": self.executed_receipt_cids,
            "counterexamples": tuple(item.to_record() for item in self.counterexamples),
            "unresolved_requirement_ids": self.unresolved_requirement_ids,
            "mandatory_fallback_pending": self.mandatory_fallback_pending,
            "human_review_required": self.human_review_required,
            "policy_cid": self.policy_cid,
            "structurally_complete": self.structurally_complete,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "bundle_id": self.bundle_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationBundle:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="verification bundle",
        )
        fields = {
            "plan_cid",
            "repository_tree_cid",
            "environment_cid",
            "required_check_key_cids",
            "receipts",
            "reused_receipt_cids",
            "executed_receipt_cids",
            "counterexamples",
            "unresolved_requirement_ids",
            "mandatory_fallback_pending",
            "human_review_required",
            "policy_cid",
            "structurally_complete",
            "bundle_id",
        }
        _reject_unknown(payload, fields, artifact_name="verification bundle")
        result = cls(
            plan_cid=payload.get("plan_cid", ""),
            repository_tree_cid=payload.get("repository_tree_cid", ""),
            environment_cid=payload.get("environment_cid", ""),
            required_check_key_cids=tuple(
                payload.get("required_check_key_cids") or ()
            ),
            receipts=tuple(payload.get("receipts") or ()),
            reused_receipt_cids=tuple(payload.get("reused_receipt_cids") or ()),
            executed_receipt_cids=tuple(payload.get("executed_receipt_cids") or ()),
            counterexamples=tuple(payload.get("counterexamples") or ()),
            unresolved_requirement_ids=tuple(
                payload.get("unresolved_requirement_ids") or ()
            ),
            mandatory_fallback_pending=payload.get("mandatory_fallback_pending"),
            human_review_required=payload.get("human_review_required"),
            policy_cid=payload.get("policy_cid", ""),
        )
        if "structurally_complete" in payload and not isinstance(
            payload["structurally_complete"], bool
        ):
            raise VerificationContractError(
                "structurally_complete projection must be a boolean"
            )
        _check_projection(
            payload,
            field_name="structurally_complete",
            actual=result.structurally_complete,
        )
        _check_identity(
            payload,
            result.bundle_id,
            names=("bundle_id", "content_id"),
            artifact_name="verification bundle",
        )
        return result


def _route_decision(value: Any) -> ModelRouteDecision:
    if isinstance(value, ModelRouteDecision):
        return value
    if isinstance(value, Mapping):
        return ModelRouteDecision.from_dict(value)
    raise VerificationContractError("model_route_decision is invalid")


@dataclass(frozen=True)
class VerificationSummary(_VerificationContract):
    """Compact ContextPack-ready projection of one verification bundle."""

    SCHEMA: ClassVar[str] = VERIFICATION_SUMMARY_SCHEMA
    INTERFACE: ClassVar[str] = VERIFICATION_SUMMARY_INTERFACE

    repository_tree_cid: str
    environment_cid: str
    changed_symbol_version_cids: tuple[str, ...]
    dependency_cone_symbols: tuple[str, ...]
    selected_tests: tuple[str, ...]
    reused_check_key_cids: tuple[str, ...]
    executed_check_key_cids: tuple[str, ...]
    failure_receipt_cids: tuple[str, ...]
    counterexample_cids: tuple[str, ...]
    unresolved_obligation_cids: tuple[str, ...]
    full_suite_pending: bool
    human_review_required: bool
    verification_wall_time_ms: int
    reused_time_saved_ms: int
    counterexample_context_tokens: int
    aggregate_terminal_status: TerminalStatus
    model_route_decision: ModelRouteDecision
    policy_cid: str

    def __post_init__(self) -> None:
        for name in ("repository_tree_cid", "environment_cid", "policy_cid"):
            object.__setattr__(self, name, _cid(getattr(self, name), field_name=name))
        for name in (
            "changed_symbol_version_cids",
            "reused_check_key_cids",
            "executed_check_key_cids",
            "failure_receipt_cids",
            "counterexample_cids",
            "unresolved_obligation_cids",
        ):
            object.__setattr__(
                self, name, _cids(getattr(self, name), field_name=name)
            )
        for name in ("dependency_cone_symbols", "selected_tests"):
            object.__setattr__(
                self,
                name,
                _strings(getattr(self, name), field_name=name, item_bytes=2_048),
            )
        if set(self.reused_check_key_cids) & set(self.executed_check_key_cids):
            raise VerificationContractError(
                "summary reused and executed key sets must be disjoint"
            )
        for name in ("full_suite_pending", "human_review_required"):
            object.__setattr__(
                self, name, _boolean(getattr(self, name), field_name=name)
            )
        for name in (
            "verification_wall_time_ms",
            "reused_time_saved_ms",
            "counterexample_context_tokens",
        ):
            object.__setattr__(
                self,
                name,
                _integer(getattr(self, name), field_name=name),
            )
        object.__setattr__(
            self,
            "aggregate_terminal_status",
            _enum(
                self.aggregate_terminal_status,
                TerminalStatus,
                field_name="aggregate_terminal_status",
            ),
        )
        object.__setattr__(
            self,
            "model_route_decision",
            _route_decision(self.model_route_decision),
        )
        if self.human_review_required != self.model_route_decision.requires_human_review:
            raise VerificationContractError(
                "summary human-review flag disagrees with model route"
            )
        _bounded(
            self,
            artifact_name="verification summary",
            maximum=MAX_SUMMARY_BYTES,
        )

    @property
    def summary_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "repository_tree_cid": self.repository_tree_cid,
            "environment_cid": self.environment_cid,
            "changed_symbol_version_cids": self.changed_symbol_version_cids,
            "dependency_cone_symbols": self.dependency_cone_symbols,
            "selected_tests": self.selected_tests,
            "reused_check_key_cids": self.reused_check_key_cids,
            "executed_check_key_cids": self.executed_check_key_cids,
            "failure_receipt_cids": self.failure_receipt_cids,
            "counterexample_cids": self.counterexample_cids,
            "unresolved_obligation_cids": self.unresolved_obligation_cids,
            "full_suite_pending": self.full_suite_pending,
            "human_review_required": self.human_review_required,
            "verification_wall_time_ms": self.verification_wall_time_ms,
            "reused_time_saved_ms": self.reused_time_saved_ms,
            "counterexample_context_tokens": self.counterexample_context_tokens,
            "aggregate_terminal_status": self.aggregate_terminal_status,
            "model_route_decision": self.model_route_decision.to_record(),
            "policy_cid": self.policy_cid,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "summary_id": self.summary_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationSummary:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="verification summary",
        )
        fields = {
            "repository_tree_cid",
            "environment_cid",
            "changed_symbol_version_cids",
            "dependency_cone_symbols",
            "selected_tests",
            "reused_check_key_cids",
            "executed_check_key_cids",
            "failure_receipt_cids",
            "counterexample_cids",
            "unresolved_obligation_cids",
            "full_suite_pending",
            "human_review_required",
            "verification_wall_time_ms",
            "reused_time_saved_ms",
            "counterexample_context_tokens",
            "aggregate_terminal_status",
            "model_route_decision",
            "policy_cid",
            "summary_id",
        }
        _reject_unknown(payload, fields, artifact_name="verification summary")
        result = cls(
            repository_tree_cid=payload.get("repository_tree_cid", ""),
            environment_cid=payload.get("environment_cid", ""),
            changed_symbol_version_cids=tuple(
                payload.get("changed_symbol_version_cids") or ()
            ),
            dependency_cone_symbols=tuple(
                payload.get("dependency_cone_symbols") or ()
            ),
            selected_tests=tuple(payload.get("selected_tests") or ()),
            reused_check_key_cids=tuple(
                payload.get("reused_check_key_cids") or ()
            ),
            executed_check_key_cids=tuple(
                payload.get("executed_check_key_cids") or ()
            ),
            failure_receipt_cids=tuple(payload.get("failure_receipt_cids") or ()),
            counterexample_cids=tuple(payload.get("counterexample_cids") or ()),
            unresolved_obligation_cids=tuple(
                payload.get("unresolved_obligation_cids") or ()
            ),
            full_suite_pending=payload.get("full_suite_pending"),
            human_review_required=payload.get("human_review_required"),
            verification_wall_time_ms=payload.get("verification_wall_time_ms", -1),
            reused_time_saved_ms=payload.get("reused_time_saved_ms", -1),
            counterexample_context_tokens=payload.get(
                "counterexample_context_tokens", -1
            ),
            aggregate_terminal_status=payload.get("aggregate_terminal_status", ""),
            model_route_decision=_route_decision(payload.get("model_route_decision")),
            policy_cid=payload.get("policy_cid", ""),
        )
        _check_identity(
            payload,
            result.summary_id,
            names=("summary_id", "content_id"),
            artifact_name="verification summary",
        )
        return result


_FAIL_CLOSED_STATUS_ORDER: Final[tuple[TerminalStatus, ...]] = (
    TerminalStatus.INVALID,
    TerminalStatus.STALE,
    TerminalStatus.SIMULATED,
    TerminalStatus.CANCELLED,
    TerminalStatus.TIMEOUT,
    TerminalStatus.UNAVAILABLE,
    TerminalStatus.UNKNOWN,
    TerminalStatus.NOT_MODELED,
    TerminalStatus.DISPROVED,
    TerminalStatus.FAILED,
)


def aggregate_terminal_status(
    statuses: Iterable[TerminalStatus | str],
    *,
    unresolved_obligation_count: int = 0,
) -> TerminalStatus:
    """Return the fail-closed aggregate; it can never improve a leaf."""

    unresolved = _integer(
        unresolved_obligation_count,
        field_name="unresolved_obligation_count",
    )
    normalized = tuple(
        _enum(item, TerminalStatus, field_name="terminal status") for item in statuses
    )
    if unresolved or not normalized:
        return TerminalStatus.UNKNOWN
    for candidate in _FAIL_CLOSED_STATUS_ORDER:
        if candidate in normalized:
            return candidate
    if all(item is TerminalStatus.PROVED for item in normalized):
        return TerminalStatus.PROVED
    if all(item.successful for item in normalized):
        return TerminalStatus.PASSED
    return TerminalStatus.UNKNOWN


def _commitment_leaves(values: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise VerificationContractError("admitted_leaves must be a sequence")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise VerificationBoundsError("admitted_leaves exceeds item bound")
    leaves: list[Mapping[str, Any]] = []
    for index, value in enumerate(values):
        leaf = _mapping(value, field_name=f"admitted_leaves[{index}]", required=True)
        if set(leaf) != {"key_cid", "receipt_cid", "receipt_kind", "status"}:
            raise VerificationContractError(
                "commitment leaf must bind key, receipt, kind, and status exactly"
            )
        kind = _enum(
            leaf["receipt_kind"],
            VerificationReceiptKind,
            field_name="leaf receipt_kind",
        )
        status = _enum(leaf["status"], TerminalStatus, field_name="leaf status")
        if kind is VerificationReceiptKind.PROOF:
            if status in {TerminalStatus.PASSED, TerminalStatus.FAILED}:
                raise VerificationContractError(
                    "proof commitment leaf cannot use test/check status"
                )
        elif status in {TerminalStatus.PROVED, TerminalStatus.DISPROVED}:
            raise VerificationContractError(
                "non-proof commitment leaf cannot use proof status"
            )
        normalized = MappingProxyType(
            {
                "key_cid": _cid(leaf["key_cid"], field_name="leaf key_cid"),
                "receipt_cid": _cid(
                    leaf["receipt_cid"], field_name="leaf receipt_cid"
                ),
                "receipt_kind": kind.value,
                "status": status.value,
            }
        )
        leaves.append(normalized)
    key_ids = tuple(item["key_cid"] for item in leaves)
    receipt_ids = tuple(item["receipt_cid"] for item in leaves)
    if len(key_ids) != len(set(key_ids)) or len(receipt_ids) != len(set(receipt_ids)):
        raise VerificationContractError(
            "commitment leaves require unique key and receipt identities"
        )
    return tuple(
        sorted(leaves, key=lambda item: (item["key_cid"], item["receipt_cid"]))
    )


def _sha256_digest(data: bytes) -> bytes:
    import hashlib

    return hashlib.sha256(data).digest()


def _merkle_root(leaves: Sequence[Mapping[str, Any]]) -> str:
    if not leaves:
        digest = _sha256_digest(b"IVP-EMPTY@1\x00")
    else:
        level = [
            _sha256_digest(b"IVP-LEAF@1\x00" + canonical_json_bytes(item))
            for item in leaves
        ]
        while len(level) > 1:
            next_level: list[bytes] = []
            for index in range(0, len(level), 2):
                if index + 1 == len(level):
                    next_level.append(level[index])
                else:
                    next_level.append(
                        _sha256_digest(
                            b"IVP-NODE@1\x00" + level[index] + level[index + 1]
                        )
                    )
            level = next_level
        digest = level[0]
    return "sha256:" + digest.hex()


@dataclass(frozen=True)
class VerificationCommitment(_VerificationContract):
    """Structural Merkle commitment over admitted verification receipts.

    This object is not a zero-knowledge proof.  A signed receipt is not proof
    of test execution unless its issuer is trusted, and structural validation
    is not cryptographic validation of the underlying execution.
    """

    SCHEMA: ClassVar[str] = VERIFICATION_COMMITMENT_SCHEMA
    INTERFACE: ClassVar[str] = VERIFICATION_COMMITMENT_INTERFACE
    IS_ZERO_KNOWLEDGE_PROOF: ClassVar[bool] = False
    HASH_ALGORITHM: ClassVar[str] = "sha2-256"
    LEAF_CODEC: ClassVar[str] = "canonical-dag-json@1"
    LEAF_DOMAIN: ClassVar[str] = "IVP-LEAF@1"
    NODE_DOMAIN: ClassVar[str] = "IVP-NODE@1"
    EMPTY_DOMAIN: ClassVar[str] = "IVP-EMPTY@1"

    repository_tree_cid: str
    environment_cid: str
    required_check_key_cids: tuple[str, ...]
    admitted_leaves: tuple[Mapping[str, Any], ...]
    public_statement: Mapping[str, Any]
    unresolved_obligation_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_tree_cid",
            _cid(self.repository_tree_cid, field_name="repository_tree_cid"),
        )
        object.__setattr__(
            self,
            "environment_cid",
            _cid(self.environment_cid, field_name="environment_cid"),
        )
        object.__setattr__(
            self,
            "required_check_key_cids",
            _cids(
                self.required_check_key_cids,
                field_name="required_check_key_cids",
                required=True,
            ),
        )
        object.__setattr__(
            self, "admitted_leaves", _commitment_leaves(self.admitted_leaves)
        )
        leaf_keys = {item["key_cid"] for item in self.admitted_leaves}
        if leaf_keys != set(self.required_check_key_cids):
            raise VerificationIdentityError(
                "commitment leaves must cover the exact required check set"
            )
        object.__setattr__(
            self,
            "public_statement",
            _mapping(
                self.public_statement,
                field_name="public_statement",
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "unresolved_obligation_count",
            _integer(
                self.unresolved_obligation_count,
                field_name="unresolved_obligation_count",
            ),
        )
        _bounded(self, artifact_name="verification commitment")

    @property
    def merkle_root(self) -> str:
        return _merkle_root(self.admitted_leaves)

    @property
    def required_check_set_cid(self) -> str:
        return _structured_cid(
            "ipfs_accelerate_py/agent-supervisor/required-verification-check-set@1",
            {"key_cids": self.required_check_key_cids},
            field_name="required_check_key_cids",
        )

    @property
    def aggregate_terminal_status(self) -> TerminalStatus:
        return aggregate_terminal_status(
            (item["status"] for item in self.admitted_leaves),
            unresolved_obligation_count=self.unresolved_obligation_count,
        )

    @property
    def commitment_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "repository_tree_cid": self.repository_tree_cid,
            "environment_cid": self.environment_cid,
            "required_check_key_cids": self.required_check_key_cids,
            "admitted_leaves": self.admitted_leaves,
            "public_statement": self.public_statement,
            "unresolved_obligation_count": self.unresolved_obligation_count,
            "merkle_root": self.merkle_root,
            "required_check_set_cid": self.required_check_set_cid,
            "aggregate_terminal_status": self.aggregate_terminal_status,
            "hash_algorithm": self.HASH_ALGORITHM,
            "leaf_codec": self.LEAF_CODEC,
            "leaf_domain": self.LEAF_DOMAIN,
            "node_domain": self.NODE_DOMAIN,
            "empty_domain": self.EMPTY_DOMAIN,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "commitment_id": self.commitment_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationCommitment:
        _check_header(
            payload,
            schema=cls.SCHEMA,
            interface=cls.INTERFACE,
            artifact_name="verification commitment",
        )
        fields = {
            "repository_tree_cid",
            "environment_cid",
            "required_check_key_cids",
            "admitted_leaves",
            "public_statement",
            "unresolved_obligation_count",
            "merkle_root",
            "required_check_set_cid",
            "aggregate_terminal_status",
            "hash_algorithm",
            "leaf_codec",
            "leaf_domain",
            "node_domain",
            "empty_domain",
            "commitment_id",
        }
        _reject_unknown(payload, fields, artifact_name="verification commitment")
        constants = {
            "hash_algorithm": cls.HASH_ALGORITHM,
            "leaf_codec": cls.LEAF_CODEC,
            "leaf_domain": cls.LEAF_DOMAIN,
            "node_domain": cls.NODE_DOMAIN,
            "empty_domain": cls.EMPTY_DOMAIN,
        }
        for name, expected in constants.items():
            if payload.get(name) != expected:
                raise VerificationContractError(
                    f"verification commitment has unsupported {name}"
                )
        result = cls(
            repository_tree_cid=payload.get("repository_tree_cid", ""),
            environment_cid=payload.get("environment_cid", ""),
            required_check_key_cids=tuple(
                payload.get("required_check_key_cids") or ()
            ),
            admitted_leaves=tuple(payload.get("admitted_leaves") or ()),
            public_statement=payload.get("public_statement") or {},
            unresolved_obligation_count=payload.get(
                "unresolved_obligation_count", -1
            ),
        )
        _check_projection(payload, field_name="merkle_root", actual=result.merkle_root)
        _check_projection(
            payload,
            field_name="required_check_set_cid",
            actual=result.required_check_set_cid,
        )
        _check_projection(
            payload,
            field_name="aggregate_terminal_status",
            actual=result.aggregate_terminal_status,
        )
        _check_identity(
            payload,
            result.commitment_id,
            names=("commitment_id", "content_id"),
            artifact_name="verification commitment",
        )
        return result


__all__ = [
    "CACHE_REUSE_DECISION_INTERFACE",
    "CACHE_REUSE_DECISION_SCHEMA",
    "COUNTEREXAMPLE_RECEIPT_INTERFACE",
    "COUNTEREXAMPLE_RECEIPT_SCHEMA",
    "MODEL_ROUTE_DECISION_INTERFACE",
    "MODEL_ROUTE_DECISION_SCHEMA",
    "PROOF_OBLIGATION_NOT_APPLICABLE_CID",
    "PROOF_RECEIPT_INTERFACE",
    "PROOF_RECEIPT_SCHEMA",
    "STATIC_ANALYSIS_RECEIPT_INTERFACE",
    "STATIC_ANALYSIS_RECEIPT_SCHEMA",
    "TERMINAL_STATUS_PRECEDENCE",
    "TEST_RECEIPT_INTERFACE",
    "TEST_RECEIPT_SCHEMA",
    "TYPE_CHECK_RECEIPT_INTERFACE",
    "TYPE_CHECK_RECEIPT_SCHEMA",
    "VERIFICATION_BUNDLE_INTERFACE",
    "VERIFICATION_BUNDLE_SCHEMA",
    "VERIFICATION_COMMITMENT_INTERFACE",
    "VERIFICATION_COMMITMENT_SCHEMA",
    "VERIFICATION_CONTRACT_VERSION",
    "VERIFICATION_PLAN_INTERFACE",
    "VERIFICATION_PLAN_SCHEMA",
    "VERIFICATION_RECEIPT_KEY_INTERFACE",
    "VERIFICATION_RECEIPT_KEY_SCHEMA",
    "VERIFICATION_SUMMARY_INTERFACE",
    "VERIFICATION_SUMMARY_SCHEMA",
    "CacheReuseDecision",
    "CacheReuseDisposition",
    "CounterexampleReceipt",
    "DiagnosticValueState",
    "DirectExecutionObservation",
    "ModelRoute",
    "ModelRouteDecision",
    "ProofReceipt",
    "StaticAnalysisReceipt",
    "TerminalStatus",
    "TestReceipt",
    "TypeCheckReceipt",
    "VerificationBoundsError",
    "VerificationBundle",
    "VerificationCommitment",
    "VerificationContractError",
    "VerificationIdentityCompiler",
    "VerificationIdentityError",
    "VerificationPlan",
    "VerificationReceipt",
    "VerificationReceiptKey",
    "VerificationReceiptKind",
    "VerificationSummary",
    "aggregate_terminal_status",
]

# Public stable spelling for future builders and conformance tests.
TERMINAL_STATUS_PRECEDENCE: Final[tuple[TerminalStatus, ...]] = (
    *_FAIL_CLOSED_STATUS_ORDER,
    TerminalStatus.PROVED,
    TerminalStatus.PASSED,
)
