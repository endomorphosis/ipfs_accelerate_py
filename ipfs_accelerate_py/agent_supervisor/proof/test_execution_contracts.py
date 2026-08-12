"""Typed contracts for proof-backed test-execution reuse (PTR-001).

Interfaces:

* ``TestLocatorKey@1`` — retrieval narrowing only; never authorizes skip.
* ``TestExecutionKey@1`` — exact reusable execution context.
* ``TestPassReceipt@1`` — admitted setup/call/teardown pass under a key.
* ``TestProofCertificate@1`` — locally verifiable certificate over a receipt.
* ``ReuseDecision@1`` — explicit ``RUN`` or ``SKIP`` only.

Authority doctrine (fail-closed, no parallel trust root):

* Absence, timeouts, unsupported capabilities, and ordinary exceptions map to
  :func:`reuse_run` / :func:`decision_from_absence` /
  :func:`decision_from_exception`.  They **cannot** coerce to ``SKIP``.
* Simulated / non-attested certificates may exercise serialization only and
  never authorize production ``SKIP``.
* Public payloads reject nonfinite values, unbounded material, private witness
  fields, malformed types, versionless artifacts, and illegal authority.
* Canonical serialization is deterministic (sorted keys, finite DAG-JSON
  profile v1 compatible encoding via formal verification contracts).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, Final, Iterable, NoReturn, Tuple

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    _enum,
    _ids,
    _mapping,
    _nonnegative_int,
    _reject_unknown_fields,
    _text,
    bounded_rejection_reason,
    canonical_json,
    canonical_json_bytes,
    content_identity,
)


# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

TEST_EXECUTION_CONTRACT_VERSION: Final = 1
SCHEMA_VERSION: Final = TEST_EXECUTION_CONTRACT_VERSION

TEST_LOCATOR_KEY_INTERFACE: Final = "TestLocatorKey@1"
TEST_EXECUTION_KEY_INTERFACE: Final = "TestExecutionKey@1"
TEST_PASS_RECEIPT_INTERFACE: Final = "TestPassReceipt@1"
TEST_PROOF_CERTIFICATE_INTERFACE: Final = "TestProofCertificate@1"
SIGNED_TEST_PASS_RECEIPT_V2_INTERFACE: Final = "SignedTestPassReceiptV2"
REUSE_DECISION_INTERFACE: Final = "ReuseDecision@1"

TEST_LOCATOR_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-locator-key@1"
)
TEST_EXECUTION_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-execution-key@1"
)
TEST_PASS_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-pass-receipt@1"
)
TEST_PROOF_CERTIFICATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-proof-certificate@1"
)
SIGNED_TEST_PASS_RECEIPT_V2_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/signed-test-pass-receipt-v2"
)
REUSE_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/reuse-decision@1"
)

# Bounds (fail closed on unbounded public artifacts).
MAX_TEXT_CHARS: Final = 4_096
MAX_NODE_ID_CHARS: Final = 2_048
MAX_REASON_CHARS: Final = 256
MAX_DIAGNOSTIC_KEYS: Final = 32
MAX_DIAGNOSTIC_VALUE_CHARS: Final = 512
MAX_SEQUENCE_ITEMS: Final = 256
MAX_COMPONENT_ENTRIES: Final = 128
MAX_PUBLIC_INPUT_KEYS: Final = 64
MAX_METADATA_KEYS: Final = 32


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestExecutionContractError(ContractValidationError):
    """Raised when a test-execution reuse contract is malformed or unsafe."""

    # Not a pytest test class.
    __test__ = False


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ReuseAction(str, Enum):
    """Closed action set for plugin/cache/supervisor boundaries.

    There is no third implicit truthy state.  Absence and exceptions must not
    collapse into ``SKIP``.
    """

    RUN = "RUN"
    SKIP = "SKIP"


class ReuseReasonCode(str, Enum):
    """Bounded reason codes for :class:`ReuseDecision`.

    ``SKIP`` is only valid with reason codes that encode a verified hit.
    All degradation / absence / error paths use ``RUN``-oriented codes.
    """

    # Authorizing skip
    PROOF_CACHE_HIT = "proof_cache_hit"

    # Mode / policy
    MODE_OFF = "mode_off"
    MODE_SHADOW = "mode_shadow"
    MODE_WRITE_ONLY = "mode_write_only"
    REUSE_DISABLED = "reuse_disabled"
    NON_REUSABLE = "non_reusable"
    ELIGIBILITY_DENIED = "eligibility_denied"

    # Absence / miss (always RUN)
    CANDIDATE_MISSING = "candidate_missing"
    CACHE_UNAVAILABLE = "cache_unavailable"
    PLUGIN_UNAVAILABLE = "plugin_unavailable"
    CID_PROVIDER_UNAVAILABLE = "cid_provider_unavailable"
    CERTIFICATE_PROVIDER_UNAVAILABLE = "certificate_provider_unavailable"
    VERIFIER_UNAVAILABLE = "verifier_unavailable"
    KEY_UNAVAILABLE = "key_unavailable"
    CIRCUIT_UNAVAILABLE = "circuit_unavailable"
    COORDINATION_UNAVAILABLE = "coordination_unavailable"

    # Integrity / trust (always RUN for production skip)
    CANDIDATE_INTEGRITY_FAILED = "candidate_integrity_failed"
    CERTIFICATE_NON_ATTESTED = "certificate_non_attested"
    CERTIFICATE_DEFERRED = "certificate_deferred"
    TRUST_POLICY_REJECTED = "trust_policy_rejected"
    ISSUER_REVOKED = "issuer_revoked"
    POLICY_MISMATCH = "policy_mismatch"
    EXECUTION_KEY_MISMATCH = "execution_key_mismatch"
    RECEIPT_MISMATCH = "receipt_mismatch"
    INCOMPLETE_TRACE = "incomplete_trace"
    INVALIDATION = "invalidation"
    EXPIRED_OR_REVOKED = "expired_or_revoked"
    MALFORMED_ARTIFACT = "malformed_artifact"
    OVER_BUDGET = "over_budget"
    PRIVATE_MATERIAL = "private_material"
    ILLEGAL_AUTHORITY = "illegal_authority"

    # Fail-open-to-run
    INTERNAL_ERROR_FAIL_OPEN_TO_RUN = "internal_error_fail_open_to_run"
    EXCEPTION_FAIL_OPEN_TO_RUN = "exception_fail_open_to_run"
    ABSENCE_FAIL_OPEN_TO_RUN = "absence_fail_open_to_run"
    UNSUPPORTED = "unsupported"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


# Reason codes that may accompany SKIP.  Everything else is RUN-only.
_SKIP_REASON_CODES: Final = frozenset({ReuseReasonCode.PROOF_CACHE_HIT})


class CertificateAuthority(str, Enum):
    """Authority class of a certificate or skip decision.

    ``NON_ATTESTED`` (including simulated ZK) never authorizes production skip.
    """

    UNKNOWN = "unknown"
    CANDIDATE = "candidate"
    NON_ATTESTED = "non_attested"
    AUTHORITATIVE = "authoritative"


class ProofBackendMode(str, Enum):
    """Proving backend mode bound into certificates."""

    CRYPTOGRAPHIC = "cryptographic"
    SIMULATED = "simulated"


class EligibilityClass(str, Enum):
    """Reuse eligibility class for an execution key."""

    PURE = "pure"
    SNAPSHOT_BOUND = "snapshot_bound"
    REPOSITORY_FOREST_BOUND = "repository_forest_bound"
    NON_REUSABLE = "non_reusable"


class PhaseOutcome(str, Enum):
    """Pytest phase outcome bits for pass receipts."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    XFAIL = "xfail"
    XPASS = "xpass"
    ERROR = "error"
    NOT_RUN = "not_run"
    INTERRUPTED = "interrupted"
    RERUN = "rerun"


_DISQUALIFYING_PHASE_OUTCOMES: Final = frozenset(
    {
        PhaseOutcome.FAIL,
        PhaseOutcome.SKIP,
        PhaseOutcome.XFAIL,
        PhaseOutcome.XPASS,
        PhaseOutcome.ERROR,
        PhaseOutcome.NOT_RUN,
        PhaseOutcome.INTERRUPTED,
        PhaseOutcome.RERUN,
    }
)


# ---------------------------------------------------------------------------
# Shared normalizers / bounds
# ---------------------------------------------------------------------------


def _raise_contract(message: str, cause: BaseException | None = None) -> NoReturn:
    """Raise module-local contract errors (never bare parent type at boundary)."""

    if cause is None:
        raise TestExecutionContractError(message)
    raise TestExecutionContractError(message) from cause


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
    preserve_order: bool = False,
    max_items: int = MAX_SEQUENCE_ITEMS,
) -> Tuple[str, ...]:
    try:
        items = _ids(
            values,
            field_name=field_name,
            required=required,
            preserve_order=preserve_order,
        )
    except ContractValidationError as exc:
        _raise_contract(str(exc), exc)
    if len(items) > max_items:
        _raise_contract(
            "%s exceeds bounded item count of %d" % (field_name, max_items)
        )
    for item in items:
        if len(item) > MAX_TEXT_CHARS:
            _raise_contract(
                "%s item exceeds bounded length of %d characters"
                % (field_name, MAX_TEXT_CHARS)
            )
    return items


def _bounded_mapping(
    value: Any,
    *,
    field_name: str,
    max_keys: int = MAX_METADATA_KEYS,
    max_value_chars: int = MAX_DIAGNOSTIC_VALUE_CHARS,
) -> Dict[str, Any]:
    if value is None:
        normalized: Dict[str, Any] = {}
    elif not isinstance(value, Mapping):
        _raise_contract("%s must be a mapping" % field_name)
        raise AssertionError("unreachable")  # for type checkers
    else:
        # Reuse formal mapping normalizer (private markers + canonical finite).
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
        _assert_finite_bounded_value(item, field_name="%s.%s" % (field_name, key))
        if isinstance(item, str) and len(item) > max_value_chars:
            _raise_contract(
                "%s value exceeds bounded length of %d characters"
                % (field_name, max_value_chars)
            )
    return normalized


def _assert_finite_bounded_value(value: Any, *, field_name: str) -> None:
    """Reject nonfinite / unbounded recursive values not caught elsewhere."""

    if isinstance(value, float):
        _raise_contract(
            "%s must not contain nonfinite or floating-point values" % field_name
        )
    if isinstance(value, Mapping):
        if len(value) > MAX_METADATA_KEYS:
            _raise_contract("%s exceeds bounded key count" % field_name)
        for key, item in value.items():
            if not isinstance(key, str):
                _raise_contract("%s keys must be strings" % field_name)
            _assert_finite_bounded_value(item, field_name="%s.%s" % (field_name, key))
        return
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_SEQUENCE_ITEMS:
            _raise_contract("%s exceeds bounded item count" % field_name)
        for index, item in enumerate(value):
            _assert_finite_bounded_value(
                item, field_name="%s[%d]" % (field_name, index)
            )
        return
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > MAX_TEXT_CHARS:
            _raise_contract("%s exceeds bounded length" % field_name)
        return
    if isinstance(value, Enum):
        return
    _raise_contract(
        "%s has unsupported value type %s" % (field_name, type(value).__name__)
    )


def _require_versioned_interface(
    payload: Mapping[str, Any],
    *,
    expected_interface: str,
    artifact_name: str,
) -> None:
    """Reject versionless and wrong-interface payloads."""

    interface = payload.get("interface")
    version = payload.get("contract_version")
    if interface in (None, "") and version in (None, ""):
        _raise_contract(
            "%s is versionless; require interface %s or contract_version %s"
            % (artifact_name, expected_interface, TEST_EXECUTION_CONTRACT_VERSION)
        )
    if interface not in (None, "", expected_interface):
        _raise_contract(
            "%s interface must be %s" % (artifact_name, expected_interface)
        )
    if version not in (None, "", TEST_EXECUTION_CONTRACT_VERSION):
        _raise_contract(
            "%s contract_version must be %s"
            % (artifact_name, TEST_EXECUTION_CONTRACT_VERSION)
        )
    # If only one is present it is fine; if both present they already matched.


def _schema_or_raise(payload: Mapping[str, Any], expected: str, *, artifact_name: str) -> None:
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


def _phase_outcome(value: Any, *, field_name: str) -> PhaseOutcome:
    try:
        return _enum(value, PhaseOutcome, field_name=field_name)
    except ContractValidationError as exc:
        _raise_contract(str(exc), exc)
        raise AssertionError("unreachable") from exc


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


def _diagnostics(value: Any) -> Dict[str, Any]:
    return _bounded_mapping(
        value,
        field_name="diagnostics",
        max_keys=MAX_DIAGNOSTIC_KEYS,
        max_value_chars=MAX_DIAGNOSTIC_VALUE_CHARS,
    )


def _component_map(value: Any, *, field_name: str) -> Dict[str, str]:
    """Normalize a bounded map of component name -> content identity string."""

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


# ---------------------------------------------------------------------------
# TestLocatorKey@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestLocatorKey(CanonicalContract):
    """Locator for candidate retrieval; never authorizes reuse by itself."""

    # Not a pytest test class.
    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = TEST_LOCATOR_KEY_SCHEMA

    repository_id: str
    package_identity: str
    node_id: str
    collection_schema_version: str = "1"
    parameter_id: str = ""
    parameter_values_cid: str = ""
    non_reusable_reason: str = ""
    selection_semantics: str = "exact_node"
    root_identity: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository_id",
            _bounded_text(self.repository_id, field_name="repository_id", required=True),
        )
        object.__setattr__(
            self,
            "package_identity",
            _bounded_text(
                self.package_identity, field_name="package_identity", required=True
            ),
        )
        object.__setattr__(
            self,
            "node_id",
            _bounded_text(
                self.node_id,
                field_name="node_id",
                required=True,
                max_chars=MAX_NODE_ID_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "collection_schema_version",
            _bounded_text(
                self.collection_schema_version,
                field_name="collection_schema_version",
                required=True,
                max_chars=64,
            ),
        )
        object.__setattr__(
            self,
            "parameter_id",
            _bounded_text(self.parameter_id, field_name="parameter_id"),
        )
        object.__setattr__(
            self,
            "parameter_values_cid",
            _bounded_text(
                self.parameter_values_cid, field_name="parameter_values_cid"
            ),
        )
        object.__setattr__(
            self,
            "non_reusable_reason",
            _bounded_text(
                self.non_reusable_reason,
                field_name="non_reusable_reason",
                max_chars=MAX_REASON_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "selection_semantics",
            _bounded_text(
                self.selection_semantics,
                field_name="selection_semantics",
                required=True,
                max_chars=128,
            ),
        )
        object.__setattr__(
            self,
            "root_identity",
            _bounded_text(self.root_identity, field_name="root_identity"),
        )
        object.__setattr__(
            self,
            "metadata",
            _bounded_mapping(self.metadata, field_name="metadata"),
        )
        if self.parameter_id and not (
            self.parameter_values_cid or self.non_reusable_reason
        ):
            raise TestExecutionContractError(
                "parameterized locator requires parameter_values_cid or "
                "non_reusable_reason"
            )

    @property
    def interface(self) -> str:
        return TEST_LOCATOR_KEY_INTERFACE

    @property
    def locator_id(self) -> str:
        return self.content_id

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": TEST_EXECUTION_CONTRACT_VERSION,
            "interface": TEST_LOCATOR_KEY_INTERFACE,
            "repository_id": self.repository_id,
            "package_identity": self.package_identity,
            "root_identity": self.root_identity,
            "node_id": self.node_id,
            "collection_schema_version": self.collection_schema_version,
            "parameter_id": self.parameter_id,
            "parameter_values_cid": self.parameter_values_cid,
            "non_reusable_reason": self.non_reusable_reason,
            "selection_semantics": self.selection_semantics,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TestLocatorKey":
        _schema_or_raise(payload, cls.SCHEMA, artifact_name="test locator key")
        _require_versioned_interface(
            payload,
            expected_interface=TEST_LOCATOR_KEY_INTERFACE,
            artifact_name="test locator key",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "repository_id",
                "package_identity",
                "root_identity",
                "node_id",
                "collection_schema_version",
                "parameter_id",
                "parameter_values_cid",
                "non_reusable_reason",
                "selection_semantics",
                "metadata",
                "content_id",
                "locator_id",
            },
            artifact_name="test locator key",
        )
        result = cls(
            repository_id=payload.get("repository_id", ""),
            package_identity=payload.get("package_identity", ""),
            node_id=payload.get("node_id", ""),
            collection_schema_version=payload.get("collection_schema_version", "1"),
            parameter_id=payload.get("parameter_id", ""),
            parameter_values_cid=payload.get("parameter_values_cid", ""),
            non_reusable_reason=payload.get("non_reusable_reason", ""),
            selection_semantics=payload.get("selection_semantics", "exact_node"),
            root_identity=payload.get("root_identity", ""),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("locator_id") or payload.get("content_id")
        if claimed and claimed != result.locator_id:
            raise TestExecutionContractError(
                "test locator key content identity does not match payload"
            )
        return result


# ---------------------------------------------------------------------------
# TestExecutionKey@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestExecutionKey(CanonicalContract):
    """Exact reusable execution context bound into receipts and certificates."""

    # Not a pytest test class.
    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = TEST_EXECUTION_KEY_SCHEMA

    locator_cid: str
    repository_forest_cid: str
    git_commit_id: str = ""
    git_tree_id: str = ""
    gitlink_state_cid: str = ""
    dirty_overlay_cid: str = ""
    test_module_cid: str = ""
    test_class_cid: str = ""
    test_function_cid: str = ""
    decorator_cids: Tuple[str, ...] = ()
    parameter_source_cid: str = ""
    test_ast_cid: str = ""
    fixture_cids: Tuple[str, ...] = ()
    conftest_closure_cid: str = ""
    hook_plugin_cids: Tuple[str, ...] = ()
    static_trace_root_cid: str = ""
    static_unknown_frontier: Tuple[str, ...] = ()
    runtime_trace_root_cid: str = ""
    runtime_completeness_policy: str = ""
    pytest_version: str = ""
    python_version: str = ""
    plugin_versions_cid: str = ""
    command_semantics_cid: str = ""
    config_cid: str = ""
    markers: Tuple[str, ...] = ()
    dependency_lock_cid: str = ""
    installed_distributions_cid: str = ""
    environment_cid: str = ""
    platform_cid: str = ""
    interpreter_abi_cid: str = ""
    hardware_capability_cid: str = ""
    external_snapshot_cids: Tuple[str, ...] = ()
    policy_cid: str = ""
    canonicalization_schema_cid: str = ""
    tracer_schema_cid: str = ""
    certificate_schema_cid: str = ""
    eligibility_class: EligibilityClass = EligibilityClass.REPOSITORY_FOREST_BOUND
    components: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "locator_cid",
            _bounded_text(self.locator_cid, field_name="locator_cid", required=True),
        )
        object.__setattr__(
            self,
            "repository_forest_cid",
            _bounded_text(
                self.repository_forest_cid,
                field_name="repository_forest_cid",
                required=True,
            ),
        )
        for name in (
            "git_commit_id",
            "git_tree_id",
            "gitlink_state_cid",
            "dirty_overlay_cid",
            "test_module_cid",
            "test_class_cid",
            "test_function_cid",
            "parameter_source_cid",
            "test_ast_cid",
            "conftest_closure_cid",
            "static_trace_root_cid",
            "runtime_trace_root_cid",
            "runtime_completeness_policy",
            "pytest_version",
            "python_version",
            "plugin_versions_cid",
            "command_semantics_cid",
            "config_cid",
            "dependency_lock_cid",
            "installed_distributions_cid",
            "environment_cid",
            "platform_cid",
            "interpreter_abi_cid",
            "hardware_capability_cid",
            "policy_cid",
            "canonicalization_schema_cid",
            "tracer_schema_cid",
            "certificate_schema_cid",
        ):
            object.__setattr__(
                self, name, _bounded_text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "decorator_cids",
            _bounded_ids(self.decorator_cids, field_name="decorator_cids"),
        )
        object.__setattr__(
            self,
            "fixture_cids",
            _bounded_ids(self.fixture_cids, field_name="fixture_cids"),
        )
        object.__setattr__(
            self,
            "hook_plugin_cids",
            _bounded_ids(self.hook_plugin_cids, field_name="hook_plugin_cids"),
        )
        object.__setattr__(
            self,
            "static_unknown_frontier",
            _bounded_ids(
                self.static_unknown_frontier,
                field_name="static_unknown_frontier",
                preserve_order=True,
            ),
        )
        object.__setattr__(
            self,
            "markers",
            _bounded_ids(self.markers, field_name="markers"),
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
            "eligibility_class",
            _safe_enum(
                self.eligibility_class,
                EligibilityClass,
                field_name="eligibility_class",
            ),
        )
        object.__setattr__(
            self,
            "components",
            _component_map(self.components, field_name="components"),
        )
        object.__setattr__(
            self,
            "metadata",
            _bounded_mapping(self.metadata, field_name="metadata"),
        )

    @property
    def interface(self) -> str:
        return TEST_EXECUTION_KEY_INTERFACE

    @property
    def execution_key_id(self) -> str:
        return self.content_id

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": TEST_EXECUTION_CONTRACT_VERSION,
            "interface": TEST_EXECUTION_KEY_INTERFACE,
            "locator_cid": self.locator_cid,
            "repository_forest_cid": self.repository_forest_cid,
            "git_commit_id": self.git_commit_id,
            "git_tree_id": self.git_tree_id,
            "gitlink_state_cid": self.gitlink_state_cid,
            "dirty_overlay_cid": self.dirty_overlay_cid,
            "test_module_cid": self.test_module_cid,
            "test_class_cid": self.test_class_cid,
            "test_function_cid": self.test_function_cid,
            "decorator_cids": list(self.decorator_cids),
            "parameter_source_cid": self.parameter_source_cid,
            "test_ast_cid": self.test_ast_cid,
            "fixture_cids": list(self.fixture_cids),
            "conftest_closure_cid": self.conftest_closure_cid,
            "hook_plugin_cids": list(self.hook_plugin_cids),
            "static_trace_root_cid": self.static_trace_root_cid,
            "static_unknown_frontier": list(self.static_unknown_frontier),
            "runtime_trace_root_cid": self.runtime_trace_root_cid,
            "runtime_completeness_policy": self.runtime_completeness_policy,
            "pytest_version": self.pytest_version,
            "python_version": self.python_version,
            "plugin_versions_cid": self.plugin_versions_cid,
            "command_semantics_cid": self.command_semantics_cid,
            "config_cid": self.config_cid,
            "markers": list(self.markers),
            "dependency_lock_cid": self.dependency_lock_cid,
            "installed_distributions_cid": self.installed_distributions_cid,
            "environment_cid": self.environment_cid,
            "platform_cid": self.platform_cid,
            "interpreter_abi_cid": self.interpreter_abi_cid,
            "hardware_capability_cid": self.hardware_capability_cid,
            "external_snapshot_cids": list(self.external_snapshot_cids),
            "policy_cid": self.policy_cid,
            "canonicalization_schema_cid": self.canonicalization_schema_cid,
            "tracer_schema_cid": self.tracer_schema_cid,
            "certificate_schema_cid": self.certificate_schema_cid,
            "eligibility_class": self.eligibility_class,
            "components": dict(self.components),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TestExecutionKey":
        _schema_or_raise(payload, cls.SCHEMA, artifact_name="test execution key")
        _require_versioned_interface(
            payload,
            expected_interface=TEST_EXECUTION_KEY_INTERFACE,
            artifact_name="test execution key",
        )
        allowed = {
            "schema",
            "contract_version",
            "interface",
            "locator_cid",
            "repository_forest_cid",
            "git_commit_id",
            "git_tree_id",
            "gitlink_state_cid",
            "dirty_overlay_cid",
            "test_module_cid",
            "test_class_cid",
            "test_function_cid",
            "decorator_cids",
            "parameter_source_cid",
            "test_ast_cid",
            "fixture_cids",
            "conftest_closure_cid",
            "hook_plugin_cids",
            "static_trace_root_cid",
            "static_unknown_frontier",
            "runtime_trace_root_cid",
            "runtime_completeness_policy",
            "pytest_version",
            "python_version",
            "plugin_versions_cid",
            "command_semantics_cid",
            "config_cid",
            "markers",
            "dependency_lock_cid",
            "installed_distributions_cid",
            "environment_cid",
            "platform_cid",
            "interpreter_abi_cid",
            "hardware_capability_cid",
            "external_snapshot_cids",
            "policy_cid",
            "canonicalization_schema_cid",
            "tracer_schema_cid",
            "certificate_schema_cid",
            "eligibility_class",
            "components",
            "metadata",
            "content_id",
            "execution_key_id",
        }
        _safe_reject_unknown_fields(
            payload, allowed, artifact_name="test execution key"
        )
        result = cls(
            locator_cid=payload.get("locator_cid", ""),
            repository_forest_cid=payload.get("repository_forest_cid", ""),
            git_commit_id=payload.get("git_commit_id", ""),
            git_tree_id=payload.get("git_tree_id", ""),
            gitlink_state_cid=payload.get("gitlink_state_cid", ""),
            dirty_overlay_cid=payload.get("dirty_overlay_cid", ""),
            test_module_cid=payload.get("test_module_cid", ""),
            test_class_cid=payload.get("test_class_cid", ""),
            test_function_cid=payload.get("test_function_cid", ""),
            decorator_cids=tuple(payload.get("decorator_cids") or ()),
            parameter_source_cid=payload.get("parameter_source_cid", ""),
            test_ast_cid=payload.get("test_ast_cid", ""),
            fixture_cids=tuple(payload.get("fixture_cids") or ()),
            conftest_closure_cid=payload.get("conftest_closure_cid", ""),
            hook_plugin_cids=tuple(payload.get("hook_plugin_cids") or ()),
            static_trace_root_cid=payload.get("static_trace_root_cid", ""),
            static_unknown_frontier=tuple(
                payload.get("static_unknown_frontier") or ()
            ),
            runtime_trace_root_cid=payload.get("runtime_trace_root_cid", ""),
            runtime_completeness_policy=payload.get(
                "runtime_completeness_policy", ""
            ),
            pytest_version=payload.get("pytest_version", ""),
            python_version=payload.get("python_version", ""),
            plugin_versions_cid=payload.get("plugin_versions_cid", ""),
            command_semantics_cid=payload.get("command_semantics_cid", ""),
            config_cid=payload.get("config_cid", ""),
            markers=tuple(payload.get("markers") or ()),
            dependency_lock_cid=payload.get("dependency_lock_cid", ""),
            installed_distributions_cid=payload.get(
                "installed_distributions_cid", ""
            ),
            environment_cid=payload.get("environment_cid", ""),
            platform_cid=payload.get("platform_cid", ""),
            interpreter_abi_cid=payload.get("interpreter_abi_cid", ""),
            hardware_capability_cid=payload.get("hardware_capability_cid", ""),
            external_snapshot_cids=tuple(
                payload.get("external_snapshot_cids") or ()
            ),
            policy_cid=payload.get("policy_cid", ""),
            canonicalization_schema_cid=payload.get(
                "canonicalization_schema_cid", ""
            ),
            tracer_schema_cid=payload.get("tracer_schema_cid", ""),
            certificate_schema_cid=payload.get("certificate_schema_cid", ""),
            eligibility_class=payload.get(
                "eligibility_class", EligibilityClass.REPOSITORY_FOREST_BOUND
            ),
            components=payload.get("components") or {},
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("execution_key_id") or payload.get("content_id")
        if claimed and claimed != result.execution_key_id:
            raise TestExecutionContractError(
                "test execution key content identity does not match payload"
            )
        return result


# ---------------------------------------------------------------------------
# TestPassReceipt@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestPassReceipt(CanonicalContract):
    """Immutable receipt of a complete eligible pass under an execution key."""

    # Not a pytest test class.
    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = TEST_PASS_RECEIPT_SCHEMA

    execution_key_cid: str
    locator_cid: str
    setup_outcome: PhaseOutcome = PhaseOutcome.PASS
    call_outcome: PhaseOutcome = PhaseOutcome.PASS
    teardown_outcome: PhaseOutcome = PhaseOutcome.PASS
    setup_duration_ms: int = 0
    call_duration_ms: int = 0
    teardown_duration_ms: int = 0
    outcome_policy_id: str = ""
    disqualifying_states: Tuple[str, ...] = ()
    static_trace_root_cid: str = ""
    runtime_trace_root_cid: str = ""
    completeness_receipt_cid: str = ""
    runner_identity: str = ""
    trust_domain: str = ""
    issuer_key_id: str = ""
    nonce: str = ""
    epoch_policy_id: str = ""
    dependency_forest_cid: str = ""
    capability_root_cid: str = ""
    schema_cid: str = ""
    policy_cid: str = ""
    admitted: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_key_cid",
            _bounded_text(
                self.execution_key_cid, field_name="execution_key_cid", required=True
            ),
        )
        object.__setattr__(
            self,
            "locator_cid",
            _bounded_text(self.locator_cid, field_name="locator_cid", required=True),
        )
        object.__setattr__(
            self,
            "setup_outcome",
            _phase_outcome(self.setup_outcome, field_name="setup_outcome"),
        )
        object.__setattr__(
            self,
            "call_outcome",
            _phase_outcome(self.call_outcome, field_name="call_outcome"),
        )
        object.__setattr__(
            self,
            "teardown_outcome",
            _phase_outcome(self.teardown_outcome, field_name="teardown_outcome"),
        )
        for name in (
            "setup_duration_ms",
            "call_duration_ms",
            "teardown_duration_ms",
        ):
            object.__setattr__(
                self,
                name,
                _safe_nonnegative_int(getattr(self, name), field_name=name),
            )
        for name in (
            "outcome_policy_id",
            "static_trace_root_cid",
            "runtime_trace_root_cid",
            "completeness_receipt_cid",
            "runner_identity",
            "trust_domain",
            "issuer_key_id",
            "nonce",
            "epoch_policy_id",
            "dependency_forest_cid",
            "capability_root_cid",
            "schema_cid",
            "policy_cid",
        ):
            object.__setattr__(
                self, name, _bounded_text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "disqualifying_states",
            _bounded_ids(
                self.disqualifying_states, field_name="disqualifying_states"
            ),
        )
        object.__setattr__(self, "admitted", _bool(self.admitted, field_name="admitted"))
        object.__setattr__(
            self,
            "metadata",
            _bounded_mapping(self.metadata, field_name="metadata"),
        )

        if self.admitted:
            for phase_name, outcome in (
                ("setup_outcome", self.setup_outcome),
                ("call_outcome", self.call_outcome),
                ("teardown_outcome", self.teardown_outcome),
            ):
                if outcome is not PhaseOutcome.PASS:
                    raise TestExecutionContractError(
                        "admitted pass receipt requires %s=pass" % phase_name
                    )
            if self.disqualifying_states:
                raise TestExecutionContractError(
                    "admitted pass receipt cannot carry disqualifying_states"
                )

    @property
    def interface(self) -> str:
        return TEST_PASS_RECEIPT_INTERFACE

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def all_phases_pass(self) -> bool:
        return (
            self.setup_outcome is PhaseOutcome.PASS
            and self.call_outcome is PhaseOutcome.PASS
            and self.teardown_outcome is PhaseOutcome.PASS
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": TEST_EXECUTION_CONTRACT_VERSION,
            "interface": TEST_PASS_RECEIPT_INTERFACE,
            "execution_key_cid": self.execution_key_cid,
            "locator_cid": self.locator_cid,
            "setup_outcome": self.setup_outcome,
            "call_outcome": self.call_outcome,
            "teardown_outcome": self.teardown_outcome,
            "setup_duration_ms": self.setup_duration_ms,
            "call_duration_ms": self.call_duration_ms,
            "teardown_duration_ms": self.teardown_duration_ms,
            "outcome_policy_id": self.outcome_policy_id,
            "disqualifying_states": list(self.disqualifying_states),
            "static_trace_root_cid": self.static_trace_root_cid,
            "runtime_trace_root_cid": self.runtime_trace_root_cid,
            "completeness_receipt_cid": self.completeness_receipt_cid,
            "runner_identity": self.runner_identity,
            "trust_domain": self.trust_domain,
            "issuer_key_id": self.issuer_key_id,
            "nonce": self.nonce,
            "epoch_policy_id": self.epoch_policy_id,
            "dependency_forest_cid": self.dependency_forest_cid,
            "capability_root_cid": self.capability_root_cid,
            "schema_cid": self.schema_cid,
            "policy_cid": self.policy_cid,
            "admitted": self.admitted,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TestPassReceipt":
        _schema_or_raise(payload, cls.SCHEMA, artifact_name="test pass receipt")
        _require_versioned_interface(
            payload,
            expected_interface=TEST_PASS_RECEIPT_INTERFACE,
            artifact_name="test pass receipt",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "execution_key_cid",
                "locator_cid",
                "setup_outcome",
                "call_outcome",
                "teardown_outcome",
                "setup_duration_ms",
                "call_duration_ms",
                "teardown_duration_ms",
                "outcome_policy_id",
                "disqualifying_states",
                "static_trace_root_cid",
                "runtime_trace_root_cid",
                "completeness_receipt_cid",
                "runner_identity",
                "trust_domain",
                "issuer_key_id",
                "nonce",
                "epoch_policy_id",
                "dependency_forest_cid",
                "capability_root_cid",
                "schema_cid",
                "policy_cid",
                "admitted",
                "metadata",
                "content_id",
                "receipt_id",
            },
            artifact_name="test pass receipt",
        )
        result = cls(
            execution_key_cid=payload.get("execution_key_cid", ""),
            locator_cid=payload.get("locator_cid", ""),
            setup_outcome=payload.get("setup_outcome", PhaseOutcome.PASS),
            call_outcome=payload.get("call_outcome", PhaseOutcome.PASS),
            teardown_outcome=payload.get("teardown_outcome", PhaseOutcome.PASS),
            setup_duration_ms=payload.get("setup_duration_ms", 0),
            call_duration_ms=payload.get("call_duration_ms", 0),
            teardown_duration_ms=payload.get("teardown_duration_ms", 0),
            outcome_policy_id=payload.get("outcome_policy_id", ""),
            disqualifying_states=tuple(payload.get("disqualifying_states") or ()),
            static_trace_root_cid=payload.get("static_trace_root_cid", ""),
            runtime_trace_root_cid=payload.get("runtime_trace_root_cid", ""),
            completeness_receipt_cid=payload.get("completeness_receipt_cid", ""),
            runner_identity=payload.get("runner_identity", ""),
            trust_domain=payload.get("trust_domain", ""),
            issuer_key_id=payload.get("issuer_key_id", ""),
            nonce=payload.get("nonce", ""),
            epoch_policy_id=payload.get("epoch_policy_id", ""),
            dependency_forest_cid=payload.get("dependency_forest_cid", ""),
            capability_root_cid=payload.get("capability_root_cid", ""),
            schema_cid=payload.get("schema_cid", ""),
            policy_cid=payload.get("policy_cid", ""),
            admitted=payload.get("admitted", True),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("receipt_id") or payload.get("content_id")
        if claimed and claimed != result.receipt_id:
            raise TestExecutionContractError(
                "test pass receipt content identity does not match payload"
            )
        return result


# ---------------------------------------------------------------------------
# SignedTestPassReceiptV2
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignedTestPassReceiptV2(CanonicalContract):
    """Public linkage from an admitted receipt to runner-attestation evidence.

    This is deliberately only a typed *reference* contract.  The authoritative
    signature, its DAG-CBOR identity, key material, and local trust-policy
    evaluation live in :mod:`testing.proof_reuse.runner_pass_attestation`.
    Keeping no signature or witness bytes here prevents a DAG-JSON cache record
    from accidentally becoming a second signing or trust authority.
    """

    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = SIGNED_TEST_PASS_RECEIPT_V2_SCHEMA

    receipt_cid: str
    execution_key_cid: str
    candidate_context_cid: str
    runner_attestation_cid: str
    trust_policy_cid: str
    signer_key_cid: str
    trust_domain: str
    key_epoch: str

    def __post_init__(self) -> None:
        for name in (
            "receipt_cid", "execution_key_cid", "candidate_context_cid",
            "runner_attestation_cid", "trust_policy_cid", "signer_key_cid",
            "trust_domain", "key_epoch",
        ):
            object.__setattr__(
                self, name, _bounded_text(getattr(self, name), field_name=name, required=True)
            )

    @property
    def interface(self) -> str:
        return SIGNED_TEST_PASS_RECEIPT_V2_INTERFACE

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": TEST_EXECUTION_CONTRACT_VERSION,
            "interface": self.interface,
            "receipt_cid": self.receipt_cid,
            "execution_key_cid": self.execution_key_cid,
            "candidate_context_cid": self.candidate_context_cid,
            "runner_attestation_cid": self.runner_attestation_cid,
            "trust_policy_cid": self.trust_policy_cid,
            "signer_key_cid": self.signer_key_cid,
            "trust_domain": self.trust_domain,
            "key_epoch": self.key_epoch,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SignedTestPassReceiptV2":
        _schema_or_raise(payload, cls.SCHEMA, artifact_name="signed test pass receipt")
        _require_versioned_interface(
            payload,
            expected_interface=SIGNED_TEST_PASS_RECEIPT_V2_INTERFACE,
            artifact_name="signed test pass receipt",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema", "contract_version", "interface", "receipt_cid",
                "execution_key_cid", "candidate_context_cid",
                "runner_attestation_cid", "trust_policy_cid", "signer_key_cid",
                "trust_domain", "key_epoch", "content_id",
            },
            artifact_name="signed test pass receipt",
        )
        result = cls(
            receipt_cid=payload.get("receipt_cid", ""),
            execution_key_cid=payload.get("execution_key_cid", ""),
            candidate_context_cid=payload.get("candidate_context_cid", ""),
            runner_attestation_cid=payload.get("runner_attestation_cid", ""),
            trust_policy_cid=payload.get("trust_policy_cid", ""),
            signer_key_cid=payload.get("signer_key_cid", ""),
            trust_domain=payload.get("trust_domain", ""),
            key_epoch=payload.get("key_epoch", ""),
        )
        claimed = payload.get("content_id")
        if claimed and claimed != result.content_id:
            raise TestExecutionContractError(
                "signed test pass receipt content identity does not match payload"
            )
        return result


# ---------------------------------------------------------------------------
# TestProofCertificate@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestProofCertificate(CanonicalContract):
    """Certificate binding a receipt to public proof inputs and authority class.

    Simulated backends are forced to :attr:`CertificateAuthority.NON_ATTESTED`
    and can never authorize production ``SKIP``.
    """

    # Not a pytest test class.
    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = TEST_PROOF_CERTIFICATE_SCHEMA

    receipt_cid: str
    execution_key_cid: str
    statement_cid: str
    circuit_cid: str
    verifying_key_cid: str
    proof_system_id: str
    proof_artifact_cid: str = ""
    proof_digest: str = ""
    backend_mode: ProofBackendMode = ProofBackendMode.CRYPTOGRAPHIC
    authority: CertificateAuthority = CertificateAuthority.AUTHORITATIVE
    issuer_id: str = ""
    policy_cid: str = ""
    epoch: str = ""
    public_inputs: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "receipt_cid",
            "execution_key_cid",
            "statement_cid",
            "circuit_cid",
            "verifying_key_cid",
            "proof_system_id",
        ):
            object.__setattr__(
                self,
                name,
                _bounded_text(getattr(self, name), field_name=name, required=True),
            )
        for name in (
            "proof_artifact_cid",
            "proof_digest",
            "issuer_id",
            "policy_cid",
            "epoch",
        ):
            object.__setattr__(
                self, name, _bounded_text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "backend_mode",
            _safe_enum(
                self.backend_mode, ProofBackendMode, field_name="backend_mode"
            ),
        )
        object.__setattr__(
            self,
            "authority",
            _safe_enum(
                self.authority, CertificateAuthority, field_name="authority"
            ),
        )
        object.__setattr__(
            self,
            "public_inputs",
            _bounded_mapping(
                self.public_inputs,
                field_name="public_inputs",
                max_keys=MAX_PUBLIC_INPUT_KEYS,
                max_value_chars=MAX_TEXT_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _bounded_mapping(self.metadata, field_name="metadata"),
        )

        # Illegal-authority: simulated / non-attested cannot claim authoritative.
        if self.backend_mode is ProofBackendMode.SIMULATED:
            if self.authority is CertificateAuthority.AUTHORITATIVE:
                raise TestExecutionContractError(
                    "illegal-authority: simulated certificate cannot be authoritative"
                )
            object.__setattr__(self, "authority", CertificateAuthority.NON_ATTESTED)
        if (
            self.authority is CertificateAuthority.AUTHORITATIVE
            and self.backend_mode is not ProofBackendMode.CRYPTOGRAPHIC
        ):
            raise TestExecutionContractError(
                "illegal-authority: authoritative certificates require cryptographic backend"
            )
        if self.authority is CertificateAuthority.UNKNOWN:
            raise TestExecutionContractError(
                "illegal-authority: certificate authority must not be unknown"
            )

    @property
    def interface(self) -> str:
        return TEST_PROOF_CERTIFICATE_INTERFACE

    @property
    def certificate_id(self) -> str:
        return self.content_id

    @property
    def can_authorize_skip(self) -> bool:
        """Whether this certificate class may authorize production SKIP."""

        return (
            self.backend_mode is ProofBackendMode.CRYPTOGRAPHIC
            and self.authority is CertificateAuthority.AUTHORITATIVE
        )

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": TEST_EXECUTION_CONTRACT_VERSION,
            "interface": TEST_PROOF_CERTIFICATE_INTERFACE,
            "receipt_cid": self.receipt_cid,
            "execution_key_cid": self.execution_key_cid,
            "statement_cid": self.statement_cid,
            "circuit_cid": self.circuit_cid,
            "verifying_key_cid": self.verifying_key_cid,
            "proof_system_id": self.proof_system_id,
            "proof_artifact_cid": self.proof_artifact_cid,
            "proof_digest": self.proof_digest,
            "backend_mode": self.backend_mode,
            "authority": self.authority,
            "issuer_id": self.issuer_id,
            "policy_cid": self.policy_cid,
            "epoch": self.epoch,
            "public_inputs": dict(self.public_inputs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TestProofCertificate":
        _schema_or_raise(payload, cls.SCHEMA, artifact_name="test proof certificate")
        _require_versioned_interface(
            payload,
            expected_interface=TEST_PROOF_CERTIFICATE_INTERFACE,
            artifact_name="test proof certificate",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "receipt_cid",
                "execution_key_cid",
                "statement_cid",
                "circuit_cid",
                "verifying_key_cid",
                "proof_system_id",
                "proof_artifact_cid",
                "proof_digest",
                "backend_mode",
                "authority",
                "issuer_id",
                "policy_cid",
                "epoch",
                "public_inputs",
                "metadata",
                "content_id",
                "certificate_id",
            },
            artifact_name="test proof certificate",
        )
        result = cls(
            receipt_cid=payload.get("receipt_cid", ""),
            execution_key_cid=payload.get("execution_key_cid", ""),
            statement_cid=payload.get("statement_cid", ""),
            circuit_cid=payload.get("circuit_cid", ""),
            verifying_key_cid=payload.get("verifying_key_cid", ""),
            proof_system_id=payload.get("proof_system_id", ""),
            proof_artifact_cid=payload.get("proof_artifact_cid", ""),
            proof_digest=payload.get("proof_digest", ""),
            backend_mode=payload.get("backend_mode", ProofBackendMode.CRYPTOGRAPHIC),
            authority=payload.get("authority", CertificateAuthority.AUTHORITATIVE),
            issuer_id=payload.get("issuer_id", ""),
            policy_cid=payload.get("policy_cid", ""),
            epoch=payload.get("epoch", ""),
            public_inputs=payload.get("public_inputs") or {},
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("certificate_id") or payload.get("content_id")
        if claimed and claimed != result.certificate_id:
            raise TestExecutionContractError(
                "test proof certificate content identity does not match payload"
            )
        return result


# ---------------------------------------------------------------------------
# ReuseDecision@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReuseDecision(CanonicalContract):
    """Explicit RUN or SKIP decision; never coerced from absence or exceptions."""

    SCHEMA: ClassVar[str] = REUSE_DECISION_SCHEMA

    action: ReuseAction
    reason_code: ReuseReasonCode
    certificate_cid: str = ""
    receipt_cid: str = ""
    validation_receipt_cid: str = ""
    authority: CertificateAuthority = CertificateAuthority.UNKNOWN
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action",
            _safe_enum(self.action, ReuseAction, field_name="action"),
        )
        object.__setattr__(
            self,
            "reason_code",
            _safe_enum(self.reason_code, ReuseReasonCode, field_name="reason_code"),
        )
        for name in ("certificate_cid", "receipt_cid", "validation_receipt_cid"):
            object.__setattr__(
                self, name, _bounded_text(getattr(self, name), field_name=name)
            )
        object.__setattr__(
            self,
            "authority",
            _safe_enum(
                self.authority, CertificateAuthority, field_name="authority"
            ),
        )
        object.__setattr__(self, "diagnostics", _diagnostics(self.diagnostics))

        if self.action is ReuseAction.SKIP:
            if not self.certificate_cid or not self.receipt_cid:
                raise TestExecutionContractError(
                    "SKIP requires certificate_cid and receipt_cid"
                )
            if self.reason_code not in _SKIP_REASON_CODES:
                raise TestExecutionContractError(
                    "SKIP requires an authorizing reason_code (proof_cache_hit)"
                )
            if self.authority is not CertificateAuthority.AUTHORITATIVE:
                raise TestExecutionContractError(
                    "illegal-authority: SKIP requires authoritative certificate authority"
                )
        elif self.action is ReuseAction.RUN:
            # RUN must not smuggle skip semantics via authoritative hit fields alone.
            if self.reason_code in _SKIP_REASON_CODES:
                raise TestExecutionContractError(
                    "RUN cannot use skip-only reason_code proof_cache_hit"
                )
        else:
            # Defensive: enum should already constrain this.
            raise TestExecutionContractError(
                "decision action must be explicitly RUN or SKIP"
            )

    @property
    def interface(self) -> str:
        return REUSE_DECISION_INTERFACE

    @property
    def decision_id(self) -> str:
        return self.content_id

    @property
    def is_run(self) -> bool:
        return self.action is ReuseAction.RUN

    @property
    def is_skip(self) -> bool:
        return self.action is ReuseAction.SKIP

    def _payload(self) -> Dict[str, Any]:
        return {
            "contract_version": TEST_EXECUTION_CONTRACT_VERSION,
            "interface": REUSE_DECISION_INTERFACE,
            "action": self.action,
            "reason_code": self.reason_code,
            "certificate_cid": self.certificate_cid,
            "receipt_cid": self.receipt_cid,
            "validation_receipt_cid": self.validation_receipt_cid,
            "authority": self.authority,
            "diagnostics": dict(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReuseDecision":
        _schema_or_raise(payload, cls.SCHEMA, artifact_name="reuse decision")
        _require_versioned_interface(
            payload,
            expected_interface=REUSE_DECISION_INTERFACE,
            artifact_name="reuse decision",
        )
        _safe_reject_unknown_fields(
            payload,
            {
                "schema",
                "contract_version",
                "interface",
                "action",
                "reason_code",
                "certificate_cid",
                "receipt_cid",
                "validation_receipt_cid",
                "authority",
                "diagnostics",
                "content_id",
                "decision_id",
            },
            artifact_name="reuse decision",
        )
        # Absence of action must not coerce to SKIP.
        if "action" not in payload or payload.get("action") in (None, ""):
            raise TestExecutionContractError(
                "decision action must be explicitly RUN or SKIP; absence cannot coerce to SKIP"
            )
        raw_action = payload.get("action")
        # Normalize common wire forms while rejecting unknown.
        if isinstance(raw_action, str):
            raw_action = raw_action.strip().upper()
        result = cls(
            action=raw_action,
            reason_code=payload.get("reason_code", ReuseReasonCode.UNKNOWN),
            certificate_cid=payload.get("certificate_cid", ""),
            receipt_cid=payload.get("receipt_cid", ""),
            validation_receipt_cid=payload.get("validation_receipt_cid", ""),
            authority=payload.get("authority", CertificateAuthority.UNKNOWN),
            diagnostics=payload.get("diagnostics") or {},
        )
        claimed = payload.get("decision_id") or payload.get("content_id")
        if claimed and claimed != result.decision_id:
            raise TestExecutionContractError(
                "reuse decision content identity does not match payload"
            )
        return result


# ---------------------------------------------------------------------------
# Decision factories (absence / exceptions never become SKIP)
# ---------------------------------------------------------------------------


def reuse_run(
    reason_code: ReuseReasonCode | str = ReuseReasonCode.UNKNOWN,
    *,
    diagnostics: Mapping[str, Any] | None = None,
    authority: CertificateAuthority | str = CertificateAuthority.UNKNOWN,
) -> ReuseDecision:
    """Build an explicit RUN decision."""

    code = _safe_enum(reason_code, ReuseReasonCode, field_name="reason_code")
    if code in _SKIP_REASON_CODES:
        # Factories never elevate to SKIP; remap illegal skip reason on RUN.
        code = ReuseReasonCode.INTERNAL_ERROR_FAIL_OPEN_TO_RUN
    return ReuseDecision(
        action=ReuseAction.RUN,
        reason_code=code,
        authority=authority,
        diagnostics=dict(diagnostics or {}),
    )


def reuse_skip(
    *,
    certificate_cid: str,
    receipt_cid: str,
    validation_receipt_cid: str = "",
    reason_code: ReuseReasonCode | str = ReuseReasonCode.PROOF_CACHE_HIT,
    diagnostics: Mapping[str, Any] | None = None,
    authority: CertificateAuthority | str = CertificateAuthority.AUTHORITATIVE,
) -> ReuseDecision:
    """Build an explicit SKIP decision (authoritative only)."""

    return ReuseDecision(
        action=ReuseAction.SKIP,
        reason_code=reason_code,
        certificate_cid=certificate_cid,
        receipt_cid=receipt_cid,
        validation_receipt_cid=validation_receipt_cid,
        authority=authority,
        diagnostics=dict(diagnostics or {}),
    )


def decision_from_absence(
    reason_code: ReuseReasonCode | str = ReuseReasonCode.ABSENCE_FAIL_OPEN_TO_RUN,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> ReuseDecision:
    """Map missing optional dependency / candidate absence to RUN (never SKIP)."""

    code = _safe_enum(reason_code, ReuseReasonCode, field_name="reason_code")
    if code in _SKIP_REASON_CODES:
        code = ReuseReasonCode.ABSENCE_FAIL_OPEN_TO_RUN
    diag = dict(diagnostics or {})
    diag.setdefault("mapped_from", "absence")
    return reuse_run(code, diagnostics=diag)


def decision_from_exception(
    exc: BaseException | None = None,
    *,
    reason_code: ReuseReasonCode | str = ReuseReasonCode.EXCEPTION_FAIL_OPEN_TO_RUN,
    diagnostics: Mapping[str, Any] | None = None,
) -> ReuseDecision:
    """Map ordinary provider/cache exceptions to RUN (never SKIP).

    Exception text is not reflected into public diagnostics to avoid leaking
    secrets; only the exception type name is retained when present.
    """

    code = _safe_enum(reason_code, ReuseReasonCode, field_name="reason_code")
    if code in _SKIP_REASON_CODES:
        code = ReuseReasonCode.EXCEPTION_FAIL_OPEN_TO_RUN
    diag = dict(diagnostics or {})
    diag.setdefault("mapped_from", "exception")
    if exc is not None:
        diag.setdefault("exception_type", type(exc).__name__[:128])
    return reuse_run(code, diagnostics=diag)


def coerce_lookup_result(
    value: Any,
    *,
    on_absence: ReuseReasonCode = ReuseReasonCode.CANDIDATE_MISSING,
) -> ReuseDecision:
    """Normalize a lookup result to a typed decision without skip coercion.

    * ``ReuseDecision`` instances are returned after revalidation.
    * ``None`` / missing maps to RUN via :func:`decision_from_absence`.
    * ``BaseException`` maps to RUN via :func:`decision_from_exception`.
    * Mappings are decoded as :class:`ReuseDecision` (still reject implicit SKIP).
    * Any other type fails closed to RUN with ``malformed_artifact``.
    """

    if value is None:
        return decision_from_absence(on_absence)
    if isinstance(value, BaseException):
        return decision_from_exception(value)
    if isinstance(value, ReuseDecision):
        # Re-construct to enforce invariants.
        return ReuseDecision.from_dict(value.to_dict())
    if isinstance(value, Mapping):
        try:
            return ReuseDecision.from_dict(value)
        except (ContractValidationError, TypeError, ValueError) as exc:
            return decision_from_exception(
                exc,
                reason_code=ReuseReasonCode.MALFORMED_ARTIFACT,
                diagnostics={"stage": "coerce_lookup_result"},
            )
    return reuse_run(
        ReuseReasonCode.MALFORMED_ARTIFACT,
        diagnostics={"mapped_from": type(value).__name__},
    )


def skip_reason_for_certificate(certificate: TestProofCertificate) -> str:
    """Standard pytest skip reason prefix for a verified hit."""

    return "proof-cache-hit:%s" % certificate.certificate_id


def certificate_may_skip(certificate: TestProofCertificate) -> bool:
    """Public predicate used by policy layers before building SKIP."""

    return certificate.can_authorize_skip


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


__all__ = [
    "CertificateAuthority",
    "EligibilityClass",
    "MAX_COMPONENT_ENTRIES",
    "MAX_DIAGNOSTIC_KEYS",
    "MAX_NODE_ID_CHARS",
    "MAX_PUBLIC_INPUT_KEYS",
    "MAX_REASON_CHARS",
    "MAX_SEQUENCE_ITEMS",
    "MAX_TEXT_CHARS",
    "PhaseOutcome",
    "ProofBackendMode",
    "REUSE_DECISION_INTERFACE",
    "REUSE_DECISION_SCHEMA",
    "ReuseAction",
    "ReuseDecision",
    "ReuseReasonCode",
    "SCHEMA_VERSION",
    "SIGNED_TEST_PASS_RECEIPT_V2_INTERFACE",
    "SIGNED_TEST_PASS_RECEIPT_V2_SCHEMA",
    "TEST_EXECUTION_CONTRACT_VERSION",
    "TEST_EXECUTION_KEY_INTERFACE",
    "TEST_EXECUTION_KEY_SCHEMA",
    "TEST_LOCATOR_KEY_INTERFACE",
    "TEST_LOCATOR_KEY_SCHEMA",
    "TEST_PASS_RECEIPT_INTERFACE",
    "TEST_PASS_RECEIPT_SCHEMA",
    "TEST_PROOF_CERTIFICATE_INTERFACE",
    "TEST_PROOF_CERTIFICATE_SCHEMA",
    "TestExecutionContractError",
    "TestExecutionKey",
    "TestLocatorKey",
    "TestPassReceipt",
    "SignedTestPassReceiptV2",
    "TestProofCertificate",
    "bounded_rejection_reason",
    "canonical_json",
    "canonical_json_bytes",
    "certificate_may_skip",
    "coerce_lookup_result",
    "content_identity",
    "decision_from_absence",
    "decision_from_exception",
    "reuse_run",
    "reuse_skip",
    "skip_reason_for_certificate",
]
