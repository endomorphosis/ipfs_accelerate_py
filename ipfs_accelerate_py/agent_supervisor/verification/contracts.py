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

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final, TypeAlias, TypeVar

from ..analysis.repository_forest import (
    RepositoryForest,
    RepositoryForestError,
    forest_observation_bindings,
    freeze_repository_forest,
    replay_repository_forest,
)
from ..contract_analysis.execution_profile import (
    CapabilitySnapshot,
    ExecutionProfileError,
    LockIdentity,
    ToolIdentity,
)
from ..core.multiformats_identity import (
    MultiformatsIdentityError,
    cid_for_bytes,
    cid_for_dag_json,
    digest_hex_from_cid,
    validate_cid,
)
from ..proof.formal_verification_contracts import (
    CONTRACT_VERSION as FORMAL_VERIFICATION_CONTRACT_VERSION,
)
from ..proof.formal_verification_contracts import (
    AssuranceLevel,
    AttemptStatus,
    CanonicalContract,
    CodeProofObligation,
    ContractValidationError,
    EvidenceFreshness,
    ProofAttempt,
    ProofStage,
    ProofVerdict,
    canonical_json_bytes,
    content_identity,
)
from ..proof.formal_verification_contracts import (
    ProofReceipt as FormalProofReceipt,
)
from ..proof.test_execution_contracts import (
    TEST_EXECUTION_CONTRACT_VERSION,
    TEST_EXECUTION_KEY_INTERFACE,
    TEST_PASS_RECEIPT_INTERFACE,
    EligibilityClass,
    TestExecutionKey,
    TestPassReceipt,
)

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
_TOOL_EXECUTABLE_IDENTITY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/observed-tool-executable@1"
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

_VERSIONED_SCHEMA_RE: Final[re.Pattern[str]] = re.compile(
    r"^[^\x00\r\n]{1,512}@[1-9][0-9]*$"
)
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")
_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_OBJECT_RE: Final[re.Pattern[str]] = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REPOSITORY_TREE_OBSERVATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "repository_forest_cid",
        "git_commit_id",
        "git_tree_id",
        "gitlink_state_cid",
        "dirty_overlay_cid",
        "dirty",
        "repository_alias",
        "repository_id",
        "descriptor_cid",
        "base_repository_tree_id",
    }
)
_SANDBOX_OBSERVATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "sandbox_schema",
        "sandbox_policy",
        "filesystem_policy",
        "platform",
        "interpreter",
        "toolchain",
        "dependency_distribution",
        "environment_values",
    }
)
_TOOL_ENVIRONMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "network_policy",
        "tool_name",
        "tool_version",
        "tool_capability_name",
        "tool_launcher_identity",
        "resolved_tool_executable",
        "tool_executable_sha256",
        "tool_executable_cid",
        "tool_version_probe_argv",
        "tool_version_probe_output_cid",
        "tool_inventory_schema",
        "adapter_schema",
        "capability_environment_names",
        "capability_read_paths",
        "capability_write_paths",
        "capability_lock_identities",
        "selected_dependency_lock_path",
        "selected_dependency_lock_identity",
    }
)
_EFFECTIVE_ENVIRONMENT_FIELDS: Final[frozenset[str]] = (
    _SANDBOX_OBSERVATION_FIELDS | _TOOL_ENVIRONMENT_FIELDS
)
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
        raise VerificationBoundsError(f"{field_name} exceeds {maximum} UTF-8 bytes")
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


def _git_object_id(value: Any, *, field_name: str) -> str:
    result = _text(value, field_name=field_name, maximum=64)
    if not _GIT_OBJECT_RE.fullmatch(result):
        raise VerificationIdentityError(
            f"{field_name} must be an exact 40- or 64-hex Git object ID"
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


def _repository_tree_observation(value: Any) -> Mapping[str, Any]:
    result = _mapping(
        value,
        field_name="repository_tree_observation",
        required=True,
    )
    if set(result) != _REPOSITORY_TREE_OBSERVATION_FIELDS:
        raise VerificationContractError(
            "repository_tree_observation must contain the exact closed field set"
        )
    normalized = {
        "repository_forest_cid": _cid(
            result["repository_forest_cid"],
            field_name="repository_tree_observation.repository_forest_cid",
        ),
        "git_commit_id": _git_object_id(
            result["git_commit_id"],
            field_name="repository_tree_observation.git_commit_id",
        ),
        "git_tree_id": _git_object_id(
            result["git_tree_id"],
            field_name="repository_tree_observation.git_tree_id",
        ),
        "gitlink_state_cid": _cid(
            result["gitlink_state_cid"],
            field_name="repository_tree_observation.gitlink_state_cid",
        ),
        "dirty_overlay_cid": _cid(
            result["dirty_overlay_cid"],
            field_name="repository_tree_observation.dirty_overlay_cid",
        ),
        "dirty": _boolean(
            result["dirty"],
            field_name="repository_tree_observation.dirty",
        ),
        "repository_alias": _text(
            result["repository_alias"],
            field_name="repository_tree_observation.repository_alias",
            maximum=256,
        ),
        "repository_id": _text(
            result["repository_id"],
            field_name="repository_tree_observation.repository_id",
            maximum=2_048,
        ),
        "descriptor_cid": _cid(
            result["descriptor_cid"],
            field_name="repository_tree_observation.descriptor_cid",
        ),
        "base_repository_tree_id": _text(
            result["base_repository_tree_id"],
            field_name="repository_tree_observation.base_repository_tree_id",
            maximum=512,
        ),
    }
    return MappingProxyType(dict(sorted(normalized.items())))


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
            raise VerificationContractError(f"{field_name} must not contain duplicates")
        result.append(item)
    if required and not result:
        raise VerificationContractError(f"{field_name} must not be empty")
    return tuple(result if preserve_order else sorted(result))


def _argv(values: Any, *, field_name: str) -> tuple[str, ...]:
    """Normalize an exact argv without changing shell-free semantics."""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise VerificationContractError(f"{field_name} must be an argv sequence")
    if not values:
        raise VerificationContractError(f"{field_name} must not be empty")
    if len(values) > MAX_COLLECTION_ITEMS:
        raise VerificationBoundsError(
            f"{field_name} exceeds {MAX_COLLECTION_ITEMS} items"
        )
    result: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise VerificationContractError(f"{field_name}[{index}] must be a string")
        if "\x00" in value:
            raise VerificationContractError(
                f"{field_name}[{index}] must not contain NUL"
            )
        if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
            raise VerificationBoundsError(
                f"{field_name}[{index}] exceeds {MAX_TEXT_BYTES} UTF-8 bytes"
            )
        result.append(value)
    if not result[0].strip():
        raise VerificationContractError(f"{field_name}[0] must name an executable")
    return tuple(result)


def _sandbox_environment_observation(value: Any) -> Mapping[str, Any]:
    result = _mapping(value, field_name="observed_environment", required=True)
    if set(result) != _SANDBOX_OBSERVATION_FIELDS:
        raise VerificationContractError(
            "observed_environment must contain the exact sandbox field set"
        )
    normalized: dict[str, Any] = {
        "sandbox_schema": _versioned_schema(
            result["sandbox_schema"],
            field_name="observed_environment.sandbox_schema",
        )
    }
    for name in sorted(_SANDBOX_OBSERVATION_FIELDS - {"sandbox_schema"}):
        observation = _mapping(
            result[name],
            field_name=f"observed_environment.{name}",
            required=True,
        )
        _versioned_schema(
            observation.get("schema"),
            field_name=f"observed_environment.{name}.schema",
        )
        normalized[name] = observation
    sandbox_policy = normalized["sandbox_policy"]
    assert isinstance(sandbox_policy, Mapping)
    if set(sandbox_policy) != {
        "schema",
        "network",
        "auto_install",
        "home_cache",
        "auth_material",
    } or any(
        sandbox_policy[name] != "deny"
        for name in ("network", "auto_install", "home_cache", "auth_material")
    ):
        raise VerificationContractError(
            "observed sandbox policy must be the closed deny-by-default policy"
        )
    filesystem_policy = normalized["filesystem_policy"]
    assert isinstance(filesystem_policy, Mapping)
    if set(filesystem_policy) != {"schema", "source", "artifacts"} or (
        filesystem_policy["source"] != "read_only"
        or filesystem_policy["artifacts"] != "private_writable"
    ):
        raise VerificationContractError(
            "observed filesystem policy must isolate source and artifacts"
        )
    return MappingProxyType(dict(sorted(normalized.items())))


def _absolute_paths(values: Any, *, field_name: str) -> tuple[str, ...]:
    paths = _strings(values, field_name=field_name, item_bytes=4_096)
    for value in paths:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts or str(path) != value:
            raise VerificationIdentityError(
                f"{field_name} must contain normalized absolute paths"
            )
    return paths


def _capability_lock_identities(value: Any) -> Mapping[str, str]:
    identities = _mapping(
        value,
        field_name="environment_observation.capability_lock_identities",
    )
    normalized: dict[str, str] = {}
    for raw_path, identity in identities.items():
        path = _text(
            raw_path,
            field_name="capability lock path",
            maximum=4_096,
        )
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise VerificationIdentityError(
                "capability lock paths must be repository-relative"
            )
        normalized[path] = _sha256(
            identity,
            field_name=f"capability_lock_identities[{path}]",
        )
    return MappingProxyType(dict(sorted(normalized.items())))


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


def _reason_codes(
    values: Any, *, field_name: str, required: bool = False
) -> tuple[str, ...]:
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
    if type(version) is not int or version != VERIFICATION_CONTRACT_VERSION:
        raise VerificationContractError(
            f"{artifact_name} has an unsupported contract version"
        )
    if payload.get("interface") != interface:
        raise VerificationContractError(f"{artifact_name} has an unsupported interface")


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
        raise VerificationContractError(f"{artifact_name} contains unsupported fields")


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
            return {str(key): normalized(item) for key, item in value.items()}
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
        raise VerificationContractError(f"{artifact_name} is not canonical") from exc
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


_PROOF_BACKEND_BINDING_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "plan_id",
        "step_id",
        "attempt_stage",
        "attempt_provider_id",
        "provider_id",
        "repository_id",
        "repository_tree_identity_kind",
        "repository_tree_identity",
        "translator_id",
        "solver_id",
        "kernel_id",
        "toolchain_id",
        "policy_id",
        "theorem_registry_id",
        "ast_scope_ids",
        "premise_ids",
        "tool_name",
        "tool_version",
        "tool_executable_cid",
        "required_assurance",
    }
)


def _proof_backend_binding(
    value: Any,
    *,
    required: bool,
) -> Mapping[str, Any] | None:
    if value is None:
        if required:
            raise VerificationIdentityError(
                "proof receipt key requires a typed proof backend binding"
            )
        return None
    binding = _mapping(value, field_name="proof_backend_binding", required=True)
    if set(binding) != _PROOF_BACKEND_BINDING_FIELDS:
        raise VerificationContractError(
            "proof_backend_binding must contain the exact closed field set"
        )
    if binding.get("repository_tree_identity_kind") != "git_tree":
        raise VerificationContractError(
            "proof_backend_binding requires repository_tree_identity_kind=git_tree"
        )
    normalized: dict[str, Any] = {}
    for name in _PROOF_BACKEND_BINDING_FIELDS.difference(
        {"ast_scope_ids", "premise_ids", "attempt_stage", "required_assurance"}
    ):
        normalized[name] = _text(
            binding[name],
            field_name=f"proof_backend_binding.{name}",
            required=name not in {"provider_id", "theorem_registry_id"},
            maximum=2_048,
        )
    normalized["tool_executable_cid"] = _cid(
        binding["tool_executable_cid"],
        field_name="proof_backend_binding.tool_executable_cid",
    )
    normalized["attempt_stage"] = _enum(
        binding["attempt_stage"],
        ProofStage,
        field_name="proof_backend_binding.attempt_stage",
    ).value
    normalized["required_assurance"] = _enum(
        binding["required_assurance"],
        AssuranceLevel,
        field_name="proof_backend_binding.required_assurance",
    ).value
    normalized["ast_scope_ids"] = _strings(
        binding["ast_scope_ids"],
        field_name="proof_backend_binding.ast_scope_ids",
        required=True,
    )
    normalized["premise_ids"] = _strings(
        binding["premise_ids"],
        field_name="proof_backend_binding.premise_ids",
    )
    return MappingProxyType(dict(sorted(normalized.items())))


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
    repository_tree_observation: Mapping[str, Any]
    semantic_state_root_cid: str
    affected_symbol_version_cids: tuple[str, ...]
    environment_cid: str
    environment_observation: Mapping[str, Any]
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
    proof_backend_binding: Mapping[str, Any] | None = None

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
        repository_tree_observation = _repository_tree_observation(
            self.repository_tree_observation
        )
        if (
            _structured_cid(
                _TREE_IDENTITY_INPUT_SCHEMA,
                repository_tree_observation,
                field_name="repository_tree_observation",
            )
            != self.repository_tree_cid
        ):
            raise VerificationIdentityError(
                "repository tree observation does not match receipt key tree CID"
            )
        object.__setattr__(
            self,
            "repository_tree_observation",
            repository_tree_observation,
        )
        environment_observation = _mapping(
            self.environment_observation,
            field_name="environment_observation",
            required=True,
        )
        if set(environment_observation) != _EFFECTIVE_ENVIRONMENT_FIELDS:
            raise VerificationContractError(
                "environment_observation must contain the exact closed field set"
            )
        _sandbox_environment_observation(
            {
                name: environment_observation[name]
                for name in _SANDBOX_OBSERVATION_FIELDS
            }
        )
        _strings(
            environment_observation["capability_environment_names"],
            field_name="environment_observation.capability_environment_names",
            item_bytes=256,
        )
        _absolute_paths(
            environment_observation["capability_read_paths"],
            field_name="environment_observation.capability_read_paths",
        )
        _absolute_paths(
            environment_observation["capability_write_paths"],
            field_name="environment_observation.capability_write_paths",
        )
        capability_lock_identities = _capability_lock_identities(
            environment_observation["capability_lock_identities"]
        )
        selected_lock_path = _text(
            environment_observation["selected_dependency_lock_path"],
            field_name="environment_observation.selected_dependency_lock_path",
            maximum=4_096,
        )
        selected_path = PurePosixPath(selected_lock_path)
        if (
            selected_path.is_absolute()
            or selected_path == PurePosixPath(".")
            or ".." in selected_path.parts
            or str(selected_path) != selected_lock_path
            or selected_lock_path not in capability_lock_identities
        ):
            raise VerificationIdentityError(
                "selected dependency lock must name an observed repository-relative lock"
            )
        selected_lock_payload = environment_observation.get(
            "selected_dependency_lock_identity"
        )
        if not isinstance(selected_lock_payload, Mapping):
            raise VerificationContractError(
                "selected dependency lock identity must be a reviewed mapping"
            )
        try:
            selected_lock_identity = LockIdentity.from_dict(selected_lock_payload)
        except ExecutionProfileError as exc:
            raise VerificationContractError(
                "selected dependency lock identity is invalid"
            ) from exc
        if (
            selected_lock_identity.path != selected_lock_path
            or selected_lock_identity.identity
            != capability_lock_identities[selected_lock_path]
        ):
            raise VerificationIdentityError(
                "selected dependency lock disagrees with the reviewed lock identity"
            )
        try:
            dependency_lock_sha256 = "sha256:" + digest_hex_from_cid(
                self.dependency_lock_cid,
                codecs=("raw",),
            )
        except MultiformatsIdentityError as exc:
            raise VerificationIdentityError(
                "dependency lock CID must be an exact raw-content identity"
            ) from exc
        if capability_lock_identities[selected_lock_path] != dependency_lock_sha256:
            raise VerificationIdentityError(
                "dependency lock CID does not match the observed lock inventory"
            )
        if (
            _structured_cid(
                _ENVIRONMENT_IDENTITY_INPUT_SCHEMA,
                environment_observation,
                field_name="environment_observation",
            )
            != self.environment_cid
        ):
            raise VerificationIdentityError(
                "environment observation does not match receipt key environment CID"
            )
        object.__setattr__(self, "environment_observation", environment_observation)
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
            self,
            "tool_name",
            _text(self.tool_name, field_name="tool_name", maximum=256),
        )
        object.__setattr__(
            self,
            "tool_version",
            _text(self.tool_version, field_name="tool_version", maximum=256),
        )
        object.__setattr__(
            self,
            "network_policy",
            _token(self.network_policy, field_name="network_policy"),
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
            _enum(
                self.receipt_kind, VerificationReceiptKind, field_name="receipt_kind"
            ),
        )
        object.__setattr__(
            self,
            "adapter_schema",
            _versioned_schema(self.adapter_schema, field_name="adapter_schema"),
        )
        if environment_observation.get("tool_name") != self.tool_name:
            raise VerificationIdentityError(
                "environment observation tool name does not match receipt key"
            )
        if environment_observation.get("tool_version") != self.tool_version:
            raise VerificationIdentityError(
                "environment observation tool version does not match receipt key"
            )
        if environment_observation.get("adapter_schema") != self.adapter_schema:
            raise VerificationIdentityError(
                "environment observation adapter schema does not match receipt key"
            )
        if environment_observation.get("network_policy") != self.network_policy:
            raise VerificationIdentityError(
                "environment observation network policy does not match receipt key"
            )
        executable_cid = _cid(
            environment_observation.get("tool_executable_cid"),
            field_name="environment_observation.tool_executable_cid",
        )
        executable_sha256 = _sha256(
            environment_observation.get("tool_executable_sha256"),
            field_name="environment_observation.tool_executable_sha256",
        )
        _cid(
            environment_observation.get("tool_version_probe_output_cid"),
            field_name="environment_observation.tool_version_probe_output_cid",
        )
        probe_argv = _argv(
            environment_observation.get("tool_version_probe_argv"),
            field_name="environment_observation.tool_version_probe_argv",
        )
        _versioned_schema(
            environment_observation.get("tool_inventory_schema"),
            field_name="environment_observation.tool_inventory_schema",
        )
        capability_name = _text(
            environment_observation.get("tool_capability_name"),
            field_name="environment_observation.tool_capability_name",
            maximum=256,
        )
        resolved_executable = _text(
            environment_observation.get("resolved_tool_executable"),
            field_name="environment_observation.resolved_tool_executable",
            maximum=4_096,
        )
        if not PurePosixPath(resolved_executable).is_absolute():
            raise VerificationIdentityError(
                "environment observation executable must be absolute"
            )
        if probe_argv[0] != resolved_executable:
            raise VerificationIdentityError(
                "environment observation probe does not use its executable"
            )
        launcher_payload = environment_observation.get("tool_launcher_identity")
        if not isinstance(launcher_payload, Mapping):
            raise VerificationContractError(
                "environment observation tool launcher must be a mapping"
            )
        try:
            launcher = ToolIdentity.from_dict(launcher_payload)
        except ExecutionProfileError as exc:
            raise VerificationContractError(
                "environment observation tool launcher is invalid"
            ) from exc
        launcher_locator = PurePosixPath(launcher.locator)
        locator_matches = (
            str(launcher_locator) == resolved_executable
            if launcher_locator.is_absolute()
            else launcher_locator.name == PurePosixPath(resolved_executable).name
        )
        if (
            launcher.name != capability_name
            or launcher.kind != "executable"
            or launcher.identity != executable_sha256
            or not locator_matches
        ):
            raise VerificationIdentityError(
                "environment observation launcher identity is inconsistent"
            )
        expected_executable_cid = _structured_cid(
            _TOOL_EXECUTABLE_IDENTITY_SCHEMA,
            {"capability_name": capability_name, "sha256": executable_sha256},
            field_name="environment_observation.tool_executable",
        )
        if executable_cid != expected_executable_cid:
            raise VerificationIdentityError(
                "environment observation executable CID is inconsistent"
            )
        if self.receipt_kind is VerificationReceiptKind.PROOF:
            if self.proof_obligation_cid == PROOF_OBLIGATION_NOT_APPLICABLE_CID:
                raise VerificationIdentityError(
                    "proof receipts require an applicable proof obligation"
                )
            binding = _proof_backend_binding(
                self.proof_backend_binding,
                required=True,
            )
            assert binding is not None
            if (
                binding["repository_tree_identity"]
                != repository_tree_observation["git_tree_id"]
            ):
                raise VerificationIdentityError(
                    "proof backend raw Git tree does not match repository observation"
                )
            if binding["repository_id"] != repository_tree_observation["repository_id"]:
                raise VerificationIdentityError(
                    "proof backend repository does not match repository observation"
                )
            if (
                binding["tool_name"] != self.tool_name
                or binding["tool_version"] != self.tool_version
                or binding["tool_executable_cid"]
                != environment_observation["tool_executable_cid"]
            ):
                raise VerificationIdentityError(
                    "proof backend tool binding does not match the observed tool"
                )
            object.__setattr__(self, "proof_backend_binding", binding)
        elif self.proof_obligation_cid != PROOF_OBLIGATION_NOT_APPLICABLE_CID:
            raise VerificationIdentityError(
                "non-proof receipts require the canonical not-applicable obligation"
            )
        elif self.proof_backend_binding is not None:
            raise VerificationIdentityError(
                "non-proof receipts cannot carry a proof backend binding"
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
            "repository_tree_observation": self.repository_tree_observation,
            "semantic_state_root_cid": self.semantic_state_root_cid,
            "affected_symbol_version_cids": self.affected_symbol_version_cids,
            "environment_cid": self.environment_cid,
            "environment_observation": self.environment_observation,
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
            "proof_backend_binding": self.proof_backend_binding,
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
            "repository_tree_observation",
            "semantic_state_root_cid",
            "affected_symbol_version_cids",
            "environment_cid",
            "environment_observation",
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
            "proof_backend_binding",
        }
        _reject_unknown(
            payload,
            fields | {"key_id"},
            artifact_name="verification receipt key",
        )
        result = cls(
            repository_tree_cid=payload.get("repository_tree_cid", ""),
            repository_tree_observation=payload.get("repository_tree_observation")
            or {},
            semantic_state_root_cid=payload.get("semantic_state_root_cid", ""),
            affected_symbol_version_cids=tuple(
                payload.get("affected_symbol_version_cids") or ()
            ),
            environment_cid=payload.get("environment_cid", ""),
            environment_observation=payload.get("environment_observation") or {},
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
            proof_backend_binding=payload.get("proof_backend_binding"),
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
        repository_forest: RepositoryForest,
        repository_alias: str,
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
        capability_snapshot: CapabilitySnapshot,
        tool_capability_name: str,
        tool_identity: ToolIdentity,
        resolved_tool_executable: str,
        tool_executable_bytes: bytes,
        tool_version_probe_argv: Sequence[str],
        tool_version_probe_output_bytes: bytes,
        claimed_environment_cid: str,
        dependency_lock_path: str,
        dependency_lock_identity: LockIdentity,
        dependency_lock_bytes: bytes,
        selector_argv: Sequence[str],
        proof_obligation: CodeProofObligation | Mapping[str, Any] | None,
        tool_name: str,
        tool_version: str,
        configuration_bytes: bytes | None,
        fixture_data_bytes: Sequence[bytes],
        network_policy: str,
        receipt_schema_version: int,
        receipt_kind: VerificationReceiptKind | str,
        adapter_schema: str,
        proof_backend_binding: Mapping[str, Any] | None,
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

        if not isinstance(repository_forest, RepositoryForest):
            raise VerificationContractError(
                "repository_forest must be a replay-valid RepositoryForest"
            )
        if repository_forest.reason_codes:
            raise VerificationIdentityError(
                "repository forest has unresolved observation reasons"
            )
        try:
            portable_forest = freeze_repository_forest(repository_forest)
            replayed_forest = replay_repository_forest(portable_forest)
            alias = _text(
                repository_alias,
                field_name="repository_alias",
                maximum=256,
            )
            descriptor = repository_forest.descriptor_for_alias(alias)
            replayed_descriptor = replayed_forest.descriptor_for_alias(alias)
            bindings = {
                item["alias"]: item
                for item in forest_observation_bindings(replayed_forest)
            }
            binding = bindings[alias]
        except (KeyError, RepositoryForestError, TypeError, ValueError) as exc:
            raise VerificationIdentityError(
                "repository forest cannot produce an exact replayed descriptor binding"
            ) from exc
        if (
            replayed_forest.forest_id != repository_forest.forest_id
            or replayed_descriptor.descriptor_cid != descriptor.descriptor_cid
            or descriptor.reason_codes
            or not descriptor.portable_closure.gitlink_closure_complete
            or not descriptor.authority.is_writable
            or alias != repository_forest.sole_write_alias
        ):
            raise VerificationIdentityError(
                "repository forest descriptor is degraded, incomplete, or not the write root"
            )
        components = binding.get("identity_components")
        if not isinstance(components, Mapping):
            raise VerificationIdentityError(
                "repository forest descriptor lacks identity components"
            )
        tree_observation = _repository_tree_observation(
            {
                "repository_forest_cid": repository_forest.forest_id,
                "git_commit_id": components.get("commit"),
                "git_tree_id": components.get("tree"),
                "gitlink_state_cid": components.get("gitlink_closure_cid"),
                "dirty_overlay_cid": components.get("dirty_overlay_digest"),
                "dirty": components.get("dirty"),
                "repository_alias": alias,
                "repository_id": binding.get("repository_id"),
                "descriptor_cid": binding.get("descriptor_cid"),
                "base_repository_tree_id": base_ids[0],
            }
        )
        if tree_observation["base_repository_tree_id"] != base_ids[0]:
            raise VerificationIdentityError(
                "observed patched tree does not bind the agreed patch base tree"
            )
        tree_cid = _structured_cid(
            _TREE_IDENTITY_INPUT_SCHEMA,
            tree_observation,
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

        selector = _argv(selector_argv, field_name="selector_argv")
        policy = _token(network_policy, field_name="network_policy")
        sandbox_environment = _sandbox_environment_observation(observed_environment)
        if not isinstance(capability_snapshot, CapabilitySnapshot):
            raise VerificationContractError(
                "capability_snapshot must be a CapabilitySnapshot observation"
            )
        capability_name = _text(
            tool_capability_name,
            field_name="tool_capability_name",
            maximum=256,
        )
        if capability_name in capability_snapshot.unavailable_tools:
            raise VerificationIdentityError("selected verification tool is unavailable")
        executable_sha256 = _sha256(
            capability_snapshot.tool_identities.get(capability_name),
            field_name="capability_snapshot.tool_identities[selected_tool]",
        )
        if not isinstance(tool_identity, ToolIdentity):
            raise VerificationContractError(
                "tool_identity must be a reviewed ToolIdentity"
            )
        reviewed_tool = ToolIdentity.from_dict(tool_identity.to_dict())
        if (
            reviewed_tool.name != capability_name
            or reviewed_tool.kind != "executable"
            or reviewed_tool.identity != executable_sha256
        ):
            raise VerificationIdentityError(
                "reviewed tool identity does not match the capability snapshot"
            )
        executable_path = _text(
            resolved_tool_executable,
            field_name="resolved_tool_executable",
            maximum=4_096,
        )
        executable = PurePosixPath(executable_path)
        locator = PurePosixPath(reviewed_tool.locator)
        if (
            not executable.is_absolute()
            or ".." in executable.parts
            or str(executable) != executable_path
            or (
                str(locator) != executable_path
                if locator.is_absolute()
                else locator.name != executable.name
            )
        ):
            raise VerificationIdentityError(
                "resolved executable does not match the reviewed tool locator"
            )
        if type(tool_executable_bytes) is not bytes:
            raise VerificationContractError("tool_executable_bytes must be exact bytes")
        if not tool_executable_bytes:
            raise VerificationContractError("tool_executable_bytes must not be empty")
        if len(tool_executable_bytes) > MAX_RAW_IDENTITY_BYTES:
            raise VerificationBoundsError(
                f"tool_executable_bytes exceeds {MAX_RAW_IDENTITY_BYTES} bytes"
            )
        independently_observed_sha256 = (
            "sha256:" + hashlib.sha256(tool_executable_bytes).hexdigest()
        )
        if independently_observed_sha256 != executable_sha256:
            raise VerificationIdentityError(
                "resolved executable bytes do not match the capability snapshot"
            )
        if (
            capability_snapshot.network_enabled
            or capability_snapshot.auto_install_enabled
            or capability_snapshot.home_cache_enabled
            or capability_snapshot.credential_names
        ):
            raise VerificationIdentityError(
                "capability snapshot violates the hermetic verification policy"
            )
        lock_path = _text(
            dependency_lock_path,
            field_name="dependency_lock_path",
            maximum=4_096,
        )
        normalized_lock_path = PurePosixPath(lock_path)
        if (
            normalized_lock_path.is_absolute()
            or normalized_lock_path == PurePosixPath(".")
            or ".." in normalized_lock_path.parts
            or str(normalized_lock_path) != lock_path
        ):
            raise VerificationIdentityError(
                "dependency_lock_path must be normalized and repository-relative"
            )
        if type(dependency_lock_bytes) is not bytes:
            raise VerificationContractError(
                "dependency_lock_bytes must be exact observed bytes"
            )
        if len(dependency_lock_bytes) > MAX_RAW_IDENTITY_BYTES:
            raise VerificationBoundsError(
                f"dependency_lock_bytes exceeds {MAX_RAW_IDENTITY_BYTES} bytes"
            )
        observed_lock_sha256 = _sha256(
            capability_snapshot.lock_identities.get(lock_path),
            field_name="capability_snapshot.lock_identities[selected_lock]",
        )
        independently_observed_lock_sha256 = (
            "sha256:" + hashlib.sha256(dependency_lock_bytes).hexdigest()
        )
        if observed_lock_sha256 != independently_observed_lock_sha256:
            raise VerificationIdentityError(
                "dependency lock bytes do not match the capability snapshot"
            )
        if not isinstance(dependency_lock_identity, LockIdentity):
            raise VerificationContractError(
                "dependency_lock_identity must be a reviewed LockIdentity"
            )
        reviewed_lock = LockIdentity.from_dict(dependency_lock_identity.to_dict())
        if (
            reviewed_lock.path != lock_path
            or reviewed_lock.identity != observed_lock_sha256
        ):
            raise VerificationIdentityError(
                "dependency lock observation does not match the reviewed lock identity"
            )
        dependency_lock_cid = _bytes_cid(
            dependency_lock_bytes,
            field_name="dependency_lock_bytes",
        )
        declared_environment_names = set(
            sandbox_environment["environment_values"]
        ) - {"schema"}
        if declared_environment_names != set(capability_snapshot.environment_names):
            raise VerificationIdentityError(
                "sandbox environment values do not match the capability snapshot"
            )
        normalized_tool_name = _text(tool_name, field_name="tool_name", maximum=256)
        normalized_tool_version = _text(
            tool_version, field_name="tool_version", maximum=256
        )
        normalized_adapter_schema = _versioned_schema(
            adapter_schema,
            field_name="adapter_schema",
        )
        probe_argv = _argv(
            tool_version_probe_argv,
            field_name="tool_version_probe_argv",
        )
        if selector[0] != executable_path or probe_argv[0] != executable_path:
            raise VerificationIdentityError(
                "selector and version probe must use the resolved executable"
            )
        if len(selector) >= 3 and selector[1] == "-m":
            if selector[2] != normalized_tool_name:
                raise VerificationIdentityError(
                    "declared tool name does not match the selected Python module"
                )
        elif executable.name != normalized_tool_name:
            raise VerificationIdentityError(
                "declared tool name does not match the selected executable"
            )
        invocation_prefix = (
            selector[:3]
            if len(selector) >= 3 and selector[1] == "-m"
            else selector[:1]
        )
        if probe_argv[: len(invocation_prefix)] != invocation_prefix:
            raise VerificationIdentityError(
                "version probe does not use the selected command launcher/module"
            )
        executable_cid = _structured_cid(
            _TOOL_EXECUTABLE_IDENTITY_SCHEMA,
            {
                "capability_name": capability_name,
                "sha256": executable_sha256,
            },
            field_name="capability_snapshot.tool_executable",
        )
        if type(tool_version_probe_output_bytes) is not bytes:
            raise VerificationContractError(
                "tool_version_probe_output_bytes must be exact bytes"
            )
        if len(tool_version_probe_output_bytes) > 65_536:
            raise VerificationBoundsError(
                "tool_version_probe_output_bytes exceeds 65536 bytes"
            )
        try:
            probe_output_text = tool_version_probe_output_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VerificationContractError(
                "tool version probe output must be UTF-8"
            ) from exc
        version_pattern = re.compile(
            rf"(?<![A-Za-z0-9._+\-]){re.escape(normalized_tool_version)}"
            rf"(?![A-Za-z0-9._+\-])"
        )
        if (
            normalized_tool_name.casefold() not in probe_output_text.casefold()
            or version_pattern.search(probe_output_text) is None
        ):
            raise VerificationIdentityError(
                "tool version claim does not match the bounded probe output"
            )
        probe_output_cid = _bytes_cid(
            tool_version_probe_output_bytes,
            field_name="tool_version_probe_output_bytes",
        )
        environment = _mapping(
            {
                **dict(sandbox_environment),
                "network_policy": policy,
                "tool_name": normalized_tool_name,
                "tool_version": normalized_tool_version,
                "tool_capability_name": capability_name,
                "tool_launcher_identity": reviewed_tool.to_dict(),
                "resolved_tool_executable": executable_path,
                "tool_executable_sha256": executable_sha256,
                "tool_executable_cid": executable_cid,
                "tool_version_probe_argv": probe_argv,
                "tool_version_probe_output_cid": probe_output_cid,
                "tool_inventory_schema": "observed-tool-inventory@1",
                "adapter_schema": normalized_adapter_schema,
                "capability_environment_names": _strings(
                    capability_snapshot.environment_names,
                    field_name="capability_snapshot.environment_names",
                    item_bytes=256,
                ),
                "capability_read_paths": _absolute_paths(
                    capability_snapshot.read_paths,
                    field_name="capability_snapshot.read_paths",
                ),
                "capability_write_paths": _absolute_paths(
                    capability_snapshot.write_paths,
                    field_name="capability_snapshot.write_paths",
                ),
                "capability_lock_identities": _capability_lock_identities(
                    capability_snapshot.lock_identities
                ),
                "selected_dependency_lock_path": lock_path,
                "selected_dependency_lock_identity": reviewed_lock.to_dict(),
            },
            field_name="effective_environment",
            required=True,
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
            raise VerificationBoundsError("affected_symbol_versions exceeds item bound")
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

        selector_cid = _structured_cid(
            _SELECTOR_IDENTITY_INPUT_SCHEMA,
            {"argv": selector},
            field_name="selector_argv",
        )

        kind = _enum(receipt_kind, VerificationReceiptKind, field_name="receipt_kind")
        normalized_backend: Mapping[str, Any] | None = None
        if kind is VerificationReceiptKind.PROOF:
            if proof_obligation is None:
                raise VerificationIdentityError(
                    "proof receipts require an existing canonical CodeProofObligation"
                )
            obligation = _code_proof_obligation(proof_obligation)
            if not isinstance(proof_backend_binding, Mapping):
                raise VerificationIdentityError(
                    "proof receipts require a typed proof backend binding"
                )
            backend_input = dict(proof_backend_binding)
            observed_repository_id = tree_observation["repository_id"]
            claimed_repository_id = backend_input.get("repository_id")
            if claimed_repository_id not in (None, observed_repository_id):
                raise VerificationIdentityError(
                    "proof backend repository does not match the observed repository"
                )
            backend_input["repository_id"] = observed_repository_id
            claimed_required_assurance = backend_input.get("required_assurance")
            if claimed_required_assurance not in (
                None,
                obligation.required_assurance.value,
            ):
                raise VerificationIdentityError(
                    "proof backend assurance does not match the proof obligation"
                )
            backend_input["required_assurance"] = (
                obligation.required_assurance.value
            )
            backend = _proof_backend_binding(backend_input, required=True)
            assert backend is not None
            normalized_backend = backend
            if obligation.repository_tree_id != tree_observation["git_tree_id"]:
                raise VerificationIdentityError(
                    "proof obligation repository tree does not match the observed tree"
                )
            if obligation.repository_id != observed_repository_id:
                raise VerificationIdentityError(
                    "proof obligation repository does not match the observed repository"
                )
            if (
                tuple(obligation.ast_scope_ids) != tuple(backend["ast_scope_ids"])
                or tuple(obligation.premise_ids) != tuple(backend["premise_ids"])
            ):
                raise VerificationIdentityError(
                    "proof obligation scopes or premises do not match the backend binding"
                )
            proof_obligation_cid = _cid(
                obligation.obligation_id,
                field_name="proof_obligation.obligation_id",
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
            repository_tree_observation=tree_observation,
            semantic_state_root_cid=semantic_cid,
            affected_symbol_version_cids=tuple(sorted(symbol_cids)),
            environment_cid=environment_cid,
            environment_observation=environment,
            dependency_lock_cid=dependency_lock_cid,
            selector_cid=selector_cid,
            proof_obligation_cid=proof_obligation_cid,
            tool_name=normalized_tool_name,
            tool_version=normalized_tool_version,
            configuration_cid=_bytes_cid(
                configuration_bytes, field_name="configuration_bytes"
            ),
            fixture_data_cids=tuple(sorted(fixture_cids)),
            network_policy=policy,
            receipt_schema_version=receipt_schema_version,
            receipt_kind=kind,
            adapter_schema=normalized_adapter_schema,
            proof_backend_binding=normalized_backend,
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
    repository_tree_observation: Mapping[str, Any]
    environment_observation: Mapping[str, Any]
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
        tree_observation = _repository_tree_observation(
            self.repository_tree_observation
        )
        if (
            _structured_cid(
                _TREE_IDENTITY_INPUT_SCHEMA,
                tree_observation,
                field_name="repository_tree_observation",
            )
            != self.repository_tree_cid
        ):
            raise VerificationIdentityError(
                "repository tree observation does not match repository_tree_cid"
            )
        object.__setattr__(self, "repository_tree_observation", tree_observation)
        environment_observation = _mapping(
            self.environment_observation,
            field_name="environment_observation",
            required=True,
        )
        if (
            _structured_cid(
                _ENVIRONMENT_IDENTITY_INPUT_SCHEMA,
                environment_observation,
                field_name="environment_observation",
            )
            != self.environment_cid
        ):
            raise VerificationIdentityError(
                "environment observation does not match environment_cid"
            )
        object.__setattr__(self, "environment_observation", environment_observation)
        object.__setattr__(
            self,
            "terminal_status",
            _enum(self.terminal_status, TerminalStatus, field_name="terminal_status"),
        )
        object.__setattr__(
            self,
            "command_argv",
            _argv(self.command_argv, field_name="command_argv"),
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
        if self.exit_code is not None and not (
            self.stdout_artifact_cid and self.stderr_artifact_cid
        ):
            raise VerificationContractError(
                "completed execution observation requires persisted stdout and stderr"
            )
        if self.terminal_status in {
            TerminalStatus.PASSED,
            TerminalStatus.PROVED,
            TerminalStatus.DISPROVED,
        }:
            if self.exit_code != 0:
                raise VerificationContractError(
                    "conclusive execution observation requires exit_code zero"
                )
            if not (self.stdout_artifact_cid and self.stderr_artifact_cid):
                raise VerificationContractError(
                    "conclusive execution observation requires persisted output evidence"
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
            "repository_tree_observation": self.repository_tree_observation,
            "environment_observation": self.environment_observation,
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
            "repository_tree_observation",
            "environment_observation",
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
        _reject_unknown(payload, fields, artifact_name="direct execution observation")
        result = cls(
            receipt_key_cid=payload.get("receipt_key_cid", ""),
            repository_tree_cid=payload.get("repository_tree_cid", ""),
            environment_cid=payload.get("environment_cid", ""),
            repository_tree_observation=payload.get("repository_tree_observation")
            or {},
            environment_observation=payload.get("environment_observation") or {},
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
    observed_selector_cid = _structured_cid(
        _SELECTOR_IDENTITY_INPUT_SCHEMA,
        {"argv": execution.command_argv},
        field_name="execution.command_argv",
    )
    if observed_selector_cid != key.selector_cid:
        raise VerificationIdentityError(
            "execution command argv does not match the receipt selector"
        )
    inventory = execution.environment_observation
    if inventory.get("tool_name") != key.tool_name:
        raise VerificationIdentityError(
            "execution environment tool name does not match the receipt key"
        )
    if inventory.get("tool_version") != key.tool_version:
        raise VerificationIdentityError(
            "execution environment tool version does not match the receipt key"
        )
    if inventory.get("network_policy") != key.network_policy:
        raise VerificationIdentityError(
            "execution environment network policy does not match the receipt key"
        )


def _validate_receipt_kind(
    key: VerificationReceiptKey,
    expected: VerificationReceiptKind,
) -> None:
    if key.receipt_kind is not expected:
        raise VerificationContractError(f"receipt requires key kind {expected.value}")


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
        object.__setattr__(
            self, "artifact_cids", _receipt_artifacts(self.artifact_cids)
        )
        object.__setattr__(
            self,
            "reason_codes",
            _reason_codes(self.reason_codes, field_name="reason_codes"),
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
            {
                "key",
                "execution",
                "status",
                "artifact_cids",
                "reason_codes",
                "receipt_id",
            },
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
        object.__setattr__(
            self, "artifact_cids", _receipt_artifacts(self.artifact_cids)
        )
        object.__setattr__(
            self,
            "reason_codes",
            _reason_codes(self.reason_codes, field_name="reason_codes"),
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
            {
                "key",
                "execution",
                "status",
                "artifact_cids",
                "reason_codes",
                "receipt_id",
            },
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


def _strict_upstream_record(
    value: Any,
    contract_type: type[TContract],
    *,
    field_name: str,
    contract_version: int,
    interface: str | None = None,
) -> TContract:
    """Clone one upstream contract through an exact versioned public record.

    Older decoders intentionally accept versionless compatibility payloads.
    Verification admission cannot: wrappers require the exact current schema,
    integer version, and interface before delegating to those decoders.
    """

    if isinstance(value, contract_type):
        payload = value.to_record()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise VerificationContractError(
            f"{field_name} must be an existing canonical {contract_type.__name__}"
        )
    if payload.get("schema") != contract_type.SCHEMA:
        raise VerificationContractError(
            f"{field_name} requires exact schema {contract_type.SCHEMA}"
        )
    version = payload.get("contract_version")
    if type(version) is not int or version != contract_version:
        raise VerificationContractError(
            f"{field_name} requires exact integer contract_version {contract_version}"
        )
    if interface is not None and payload.get("interface") != interface:
        raise VerificationContractError(
            f"{field_name} requires exact interface {interface}"
        )
    try:
        result = contract_type.from_dict(payload)  # type: ignore[attr-defined]
    except ContractValidationError as exc:
        raise VerificationContractError(f"{field_name} is invalid") from exc
    assert isinstance(result, contract_type)
    return result


def _code_proof_obligation(value: Any) -> CodeProofObligation:
    payload = value.to_record() if isinstance(value, CodeProofObligation) else value
    if not isinstance(payload, Mapping):
        raise VerificationContractError(
            "proof_obligation must be an existing canonical CodeProofObligation"
        )
    payload = dict(payload)
    _reject_unknown(
        payload,
        {
            "repository_id",
            "repository_tree_id",
            "ast_scope_ids",
            "statement",
            "premise_ids",
            "template_id",
            "template_version",
            "template_semantic_hash",
            "invariant_class",
            "task_id",
            "required_assurance",
            "fallback_checks",
            "metadata",
            "obligation_id",
            "content_id",
        },
        artifact_name="code proof obligation",
    )
    payload["metadata"] = _mapping(
        payload.get("metadata") or {},
        field_name="proof_obligation.metadata",
    )
    result = _strict_upstream_record(
        payload,
        CodeProofObligation,
        field_name="proof_obligation",
        contract_version=FORMAL_VERIFICATION_CONTRACT_VERSION,
    )
    object.__setattr__(
        result,
        "metadata",
        _mapping(result.metadata, field_name="proof_obligation.metadata"),
    )
    return result


def _test_pass_receipt(value: Any) -> TestPassReceipt:
    result = _strict_upstream_record(
        value,
        TestPassReceipt,
        field_name="test_pass_receipt",
        contract_version=TEST_EXECUTION_CONTRACT_VERSION,
        interface=TEST_PASS_RECEIPT_INTERFACE,
    )
    object.__setattr__(
        result,
        "metadata",
        _mapping(result.metadata, field_name="test_pass_receipt.metadata"),
    )
    return result


def _test_execution_key(value: Any) -> TestExecutionKey:
    result = _strict_upstream_record(
        value,
        TestExecutionKey,
        field_name="test_execution_key",
        contract_version=TEST_EXECUTION_CONTRACT_VERSION,
        interface=TEST_EXECUTION_KEY_INTERFACE,
    )
    object.__setattr__(
        result,
        "components",
        _mapping(result.components, field_name="test_execution_key.components"),
    )
    object.__setattr__(
        result,
        "metadata",
        _mapping(result.metadata, field_name="test_execution_key.metadata"),
    )
    return result


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
            if not {
                source_key.execution_key_id,
                source_receipt.receipt_id,
            }.issubset(set(execution.artifact_cids)):
                raise VerificationIdentityError(
                    "direct execution observation does not name the test key "
                    "and pass receipt artifacts"
                )
            observed_tree = execution.repository_tree_observation
            # These are the repository fields which both contracts represent in
            # the same identity domain.  In particular, the upstream dirty
            # overlay, command, lock, fixture, config, and environment roots are
            # domain-separated projections assembled from normalized pytest
            # inputs; they must not be equated to this package's raw-byte/argv
            # cache-key components.
            tree_bindings = {
                name: (getattr(source_key, name), observed_tree.get(name))
                for name in (
                    "repository_forest_cid",
                    "git_commit_id",
                    "git_tree_id",
                    "gitlink_state_cid",
                )
            }
            if any(actual != expected for actual, expected in tree_bindings.values()):
                raise VerificationIdentityError(
                    "existing TestExecutionKey repository identity does not match "
                    "the observed repository tree"
                )
            source_descriptor_cid = source_key.components.get(
                "repository_descriptor"
            )
            if (
                not isinstance(source_descriptor_cid, str)
                or source_descriptor_cid
                != observed_tree.get("descriptor_cid")
            ):
                raise VerificationIdentityError(
                    "existing TestExecutionKey repository descriptor does not match "
                    "the observed repository write descriptor"
                )
            if self.key.tool_name != "pytest" or (
                source_key.pytest_version != self.key.tool_version
            ):
                raise VerificationIdentityError(
                    "existing TestExecutionKey pytest version does not match "
                    "the observed pytest tool"
                )
            receipt_bindings = {
                "dependency_forest_cid": (
                    source_receipt.dependency_forest_cid,
                    source_key.repository_forest_cid,
                ),
                "static_trace_root_cid": (
                    source_receipt.static_trace_root_cid,
                    source_key.static_trace_root_cid,
                ),
                "runtime_trace_root_cid": (
                    source_receipt.runtime_trace_root_cid,
                    source_key.runtime_trace_root_cid,
                ),
                "policy_cid": (source_receipt.policy_cid, source_key.policy_cid),
            }
            if any(
                not actual or not expected or actual != expected
                for actual, expected in receipt_bindings.values()
            ):
                raise VerificationIdentityError(
                    "TestPassReceipt does not bind the source key's forest, "
                    "trace, and policy identities"
                )
            if not source_receipt.completeness_receipt_cid:
                raise VerificationIdentityError(
                    "TestPassReceipt lacks a runtime completeness receipt"
                )
            if execution.terminal_status is not TerminalStatus.PASSED:
                raise VerificationIdentityError(
                    "existing passed-test bridge disagrees with direct observation"
                )
        object.__setattr__(
            self, "artifact_cids", _receipt_artifacts(self.artifact_cids)
        )
        object.__setattr__(
            self,
            "reason_codes",
            _reason_codes(self.reason_codes, field_name="reason_codes"),
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
            test_pass_receipt=_test_pass_receipt(source)
            if source is not None
            else None,
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
    result = _strict_upstream_record(
        value,
        FormalProofReceipt,
        field_name="formal_proof_receipt",
        contract_version=FORMAL_VERIFICATION_CONTRACT_VERSION,
    )
    frozen_evidence = []
    for index, evidence in enumerate(result.evidence):
        object.__setattr__(
            evidence,
            "metadata",
            _mapping(
                evidence.metadata,
                field_name=f"formal_proof_receipt.evidence[{index}].metadata",
            ),
        )
        frozen_evidence.append(evidence)
    object.__setattr__(result, "evidence", tuple(frozen_evidence))
    object.__setattr__(
        result,
        "resource_usage",
        _mapping(
            result.resource_usage,
            field_name="formal_proof_receipt.resource_usage",
        ),
    )
    object.__setattr__(
        result,
        "metadata",
        _mapping(result.metadata, field_name="formal_proof_receipt.metadata"),
    )
    return result


def _formal_proof_attempt(value: Any) -> ProofAttempt:
    result = _strict_upstream_record(
        value,
        ProofAttempt,
        field_name="proof_attempt",
        contract_version=FORMAL_VERIFICATION_CONTRACT_VERSION,
    )
    frozen_evidence = []
    for index, evidence in enumerate(result.evidence):
        object.__setattr__(
            evidence,
            "metadata",
            _mapping(
                evidence.metadata,
                field_name=f"proof_attempt.evidence[{index}].metadata",
            ),
        )
        frozen_evidence.append(evidence)
    object.__setattr__(result, "evidence", tuple(frozen_evidence))
    object.__setattr__(
        result,
        "resource_usage",
        _mapping(result.resource_usage, field_name="proof_attempt.resource_usage"),
    )
    object.__setattr__(
        result,
        "metadata",
        _mapping(result.metadata, field_name="proof_attempt.metadata"),
    )
    return result


def _formal_proof_status(
    receipt: FormalProofReceipt,
    required_assurance: AssuranceLevel | str,
) -> TerminalStatus:
    if receipt.freshness is not EvidenceFreshness.CURRENT:
        return TerminalStatus.STALE
    if any(item.simulated for item in receipt.evidence):
        return TerminalStatus.SIMULATED
    # Any conclusive non-success from stronger independent evidence must
    # dominate a provider's declaration and weaker accepted solver evidence.
    authoritative = receipt.authoritative_verdict
    if authoritative is ProofVerdict.DISPROVED:
        return TerminalStatus.DISPROVED
    if authoritative is ProofVerdict.CANCELLED:
        return TerminalStatus.CANCELLED
    if authoritative is ProofVerdict.UNSUPPORTED:
        return TerminalStatus.UNAVAILABLE
    if authoritative is ProofVerdict.ERROR:
        return TerminalStatus.INVALID
    if (
        receipt.verdict is ProofVerdict.PROVED
        and authoritative in {ProofVerdict.INCONCLUSIVE, ProofVerdict.PROVED}
        and receipt.authoritative_assurance.satisfies(
            _enum(
                required_assurance,
                AssuranceLevel,
                field_name="proof_backend_binding.required_assurance",
            )
        )
    ):
        return TerminalStatus.PROVED
    if receipt.verdict is ProofVerdict.CANCELLED:
        return TerminalStatus.CANCELLED
    if receipt.verdict is ProofVerdict.UNSUPPORTED:
        return TerminalStatus.UNAVAILABLE
    if receipt.verdict is ProofVerdict.ERROR:
        return TerminalStatus.INVALID
    return TerminalStatus.UNKNOWN


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
    proof_attempt: ProofAttempt | None = None
    artifact_cids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _key(self.key))
        _validate_receipt_kind(self.key, VerificationReceiptKind.PROOF)
        execution = _observation(self.execution)
        object.__setattr__(self, "execution", execution)
        _validate_execution_binding(self.key, execution)
        if self.formal_proof_receipt is None:
            if self.proof_attempt is not None:
                raise VerificationContractError(
                    "proof attempt cannot be authoritative without a formal receipt"
                )
            _direct_check_status(execution.terminal_status, proof=True)
        else:
            formal = _formal_proof_receipt(self.formal_proof_receipt)
            object.__setattr__(self, "formal_proof_receipt", formal)
            attempt = (
                _formal_proof_attempt(self.proof_attempt)
                if self.proof_attempt is not None
                else None
            )
            object.__setattr__(self, "proof_attempt", attempt)
            backend = self.key.proof_backend_binding
            assert backend is not None
            raw_git_tree = self.key.repository_tree_observation["git_tree_id"]
            if formal.repository_tree_id != raw_git_tree:
                raise VerificationIdentityError(
                    "formal proof receipt uses a different repository tree"
                )
            if formal.obligation_id != self.key.proof_obligation_cid:
                raise VerificationIdentityError(
                    "formal proof receipt uses a different proof obligation"
                )
            expected_formal_fields = {
                "repository_id": formal.repository_id,
                "plan_id": formal.plan_id,
                "translator_id": formal.translator_id,
                "solver_id": formal.solver_id,
                "kernel_id": formal.kernel_id,
                "toolchain_id": formal.toolchain_id,
                "policy_id": formal.policy_id,
                "theorem_registry_id": formal.theorem_registry_id,
                "ast_scope_ids": formal.ast_scope_ids,
                "premise_ids": formal.premise_ids,
                "provider_id": formal.provider_id,
                "repository_tree_identity": formal.repository_tree_id,
            }
            if any(
                expected_formal_fields[name] != backend[name]
                for name in expected_formal_fields
            ):
                raise VerificationIdentityError(
                    "formal proof receipt does not match the pre-execution backend binding"
                )
            required_artifacts = {formal.receipt_id}
            if attempt is not None:
                if formal.attempt_id != attempt.attempt_id:
                    raise VerificationIdentityError(
                        "formal proof receipt does not bind the supplied ProofAttempt"
                    )
                expected_attempt_fields = {
                    "plan_id": backend["plan_id"],
                    "step_id": backend["step_id"],
                    "obligation_id": self.key.proof_obligation_cid,
                    "repository_tree_id": raw_git_tree,
                    "provider_id": backend["attempt_provider_id"],
                }
                if any(
                    getattr(attempt, name) != expected
                    for name, expected in expected_attempt_fields.items()
                ):
                    raise VerificationIdentityError(
                        "ProofAttempt does not match the documented proof input binding"
                    )
                if attempt.status is not AttemptStatus.SUCCEEDED:
                    raise VerificationIdentityError(
                        "conclusive formal proof receipt requires a succeeded ProofAttempt"
                    )
                if attempt.stage.value != backend["attempt_stage"]:
                    raise VerificationIdentityError(
                        "ProofAttempt stage does not match the proof backend binding"
                    )
                required_artifacts.add(attempt.attempt_id)
            if not required_artifacts.issubset(set(execution.artifact_cids)):
                raise VerificationIdentityError(
                    "direct execution observation does not name the formal proof "
                    "receipt and optional attempt artifacts"
                )
            formal_status = _formal_proof_status(
                formal,
                backend["required_assurance"],
            )
            if execution.terminal_status is not formal_status:
                raise VerificationIdentityError(
                    "formal proof result conflicts with the direct execution status"
                )
        object.__setattr__(
            self, "artifact_cids", _receipt_artifacts(self.artifact_cids)
        )
        object.__setattr__(
            self,
            "reason_codes",
            _reason_codes(self.reason_codes, field_name="reason_codes"),
        )
        _bounded(self, artifact_name="proof receipt")

    @property
    def status(self) -> TerminalStatus:
        if self.formal_proof_receipt is None:
            return self.execution.terminal_status
        backend = self.key.proof_backend_binding
        assert backend is not None
        return _formal_proof_status(
            self.formal_proof_receipt,
            backend["required_assurance"],
        )

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
            "proof_attempt": (
                self.proof_attempt.to_record() if self.proof_attempt else None
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
                "proof_attempt",
                "status",
                "artifact_cids",
                "reason_codes",
                "receipt_id",
            },
            artifact_name="proof receipt",
        )
        execution = payload.get("execution")
        source = payload.get("formal_proof_receipt")
        attempt = payload.get("proof_attempt")
        result = cls(
            key=_key(payload.get("key")),
            execution=_observation(execution),
            formal_proof_receipt=(
                _formal_proof_receipt(source) if source is not None else None
            ),
            proof_attempt=(
                _formal_proof_attempt(attempt) if attempt is not None else None
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
            raise VerificationContractError(
                "source span path must be repository-relative"
            )
        start = _integer(
            span.get("start_line"), field_name="source span start", minimum=1
        )
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
            _cid(self.failed_selector, field_name="failed_selector"),
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
            _argv(self.reproduction_argv, field_name="reproduction_argv"),
        )
        object.__setattr__(
            self, "artifact_cids", _receipt_artifacts(self.artifact_cids)
        )
        object.__setattr__(
            self, "minimized", _boolean(self.minimized, field_name="minimized")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _reason_codes(self.reason_codes, field_name="reason_codes"),
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


VerificationReceipt: TypeAlias = (
    StaticAnalysisReceipt | TypeCheckReceipt | TestReceipt | ProofReceipt
)


@dataclass(frozen=True)
class CacheReuseDecision(_VerificationContract):
    """Explicit exact-key cache disposition; absence never becomes reuse."""

    SCHEMA: ClassVar[str] = CACHE_REUSE_DECISION_SCHEMA
    INTERFACE: ClassVar[str] = CACHE_REUSE_DECISION_INTERFACE

    key_cid: str
    disposition: CacheReuseDisposition
    reason_codes: tuple[str, ...]
    candidate_receipt: VerificationReceipt | None = None

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
        if self.candidate_receipt is not None:
            object.__setattr__(
                self,
                "candidate_receipt",
                _verification_receipt(self.candidate_receipt),
            )
        if self.disposition is CacheReuseDisposition.REUSED:
            if self.candidate_receipt is None:
                raise VerificationContractError(
                    "reused decision requires a complete candidate receipt"
                )
            if self.candidate_receipt.key.key_id != self.key_cid:
                raise VerificationIdentityError(
                    "reused decision candidate does not match the exact key"
                )
            if self.candidate_status not in {
                TerminalStatus.PASSED,
                TerminalStatus.PROVED,
            }:
                raise VerificationContractError(
                    "reused decision requires a successful terminal candidate"
                )
            if isinstance(self.candidate_receipt, TestReceipt):
                source_key = self.candidate_receipt.test_execution_key
                if source_key is not None and (
                    source_key.eligibility_class is EligibilityClass.NON_REUSABLE
                    or bool(source_key.components.get("non_reusable_reason"))
                ):
                    raise VerificationContractError(
                        "reused decision cannot admit a non-reusable test execution key"
                    )
        if (
            self.disposition is CacheReuseDisposition.MISSING
            and self.candidate_receipt is not None
        ):
            raise VerificationContractError(
                "missing decision cannot carry a candidate receipt"
            )
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
    def receipt_cid(self) -> str:
        return self.candidate_receipt.receipt_id if self.candidate_receipt else ""

    @property
    def candidate_status(self) -> TerminalStatus | None:
        return self.candidate_receipt.status if self.candidate_receipt else None

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
            "candidate_receipt": (
                self.candidate_receipt.to_record()
                if self.candidate_receipt is not None
                else None
            ),
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
                "candidate_receipt",
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
            candidate_receipt=(
                _verification_receipt(payload["candidate_receipt"])
                if payload.get("candidate_receipt") is not None
                else None
            ),
        )
        if "reusable" in payload and not isinstance(payload["reusable"], bool):
            raise VerificationContractError("reusable projection must be a boolean")
        _check_projection(payload, field_name="reusable", actual=result.reusable)
        _check_projection(payload, field_name="receipt_cid", actual=result.receipt_cid)
        _check_projection(
            payload,
            field_name="candidate_status",
            actual=result.candidate_status,
        )
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
        ready = sorted(
            step for step, dependencies in pending.items() if not dependencies
        )
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
    full_suite_receipt_key_cids: tuple[str, ...]
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
        decision_key_ids = tuple(item.key_cid for item in self.cache_reuse_decisions)
        if len(decision_key_ids) != len(set(decision_key_ids)):
            raise VerificationContractError(
                "cache_reuse_decisions contains more than one decision per key"
            )
        required_key_ids = {item.key_id for item in self.required_receipt_keys}
        if set(decision_key_ids) != required_key_ids:
            raise VerificationIdentityError(
                "cache_reuse_decisions must cover required receipt keys exactly"
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
        required_proof_obligations = {
            key.proof_obligation_cid
            for key in self.required_receipt_keys
            if key.receipt_kind is VerificationReceiptKind.PROOF
        }
        if set(self.affected_proof_obligation_cids) != required_proof_obligations:
            raise VerificationIdentityError(
                "affected proof obligations must equal the required proof key obligations"
            )
        object.__setattr__(
            self,
            "full_suite_receipt_key_cids",
            _cids(
                self.full_suite_receipt_key_cids,
                field_name="full_suite_receipt_key_cids",
            ),
        )
        object.__setattr__(
            self,
            "full_suite_required",
            _boolean(self.full_suite_required, field_name="full_suite_required"),
        )
        if bool(self.full_suite_receipt_key_cids) is not self.full_suite_required:
            raise VerificationContractError(
                "full-suite receipt keys must be present exactly when fallback is required"
            )
        required_keys_by_id = {
            key.key_id: key for key in self.required_receipt_keys
        }
        if not set(self.full_suite_receipt_key_cids).issubset(required_keys_by_id):
            raise VerificationIdentityError(
                "full-suite receipt keys must belong to the required check set"
            )
        if any(
            required_keys_by_id[key_id].receipt_kind
            is not VerificationReceiptKind.TEST
            for key_id in self.full_suite_receipt_key_cids
        ):
            raise VerificationContractError(
                "full-suite receipt keys must identify test receipts"
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
            raise VerificationBoundsError("step timeout exceeds maximum execution time")
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
            "full_suite_receipt_key_cids": self.full_suite_receipt_key_cids,
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
            "full_suite_receipt_key_cids",
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
            cache_reuse_decisions=tuple(payload.get("cache_reuse_decisions") or ()),
            affected_tests=tuple(payload.get("affected_tests") or ()),
            fallback_tests=tuple(payload.get("fallback_tests") or ()),
            required_static_checks=tuple(payload.get("required_static_checks") or ()),
            required_type_checks=tuple(payload.get("required_type_checks") or ()),
            affected_proof_obligation_cids=tuple(
                payload.get("affected_proof_obligation_cids") or ()
            ),
            full_suite_receipt_key_cids=tuple(
                payload.get("full_suite_receipt_key_cids") or ()
            ),
            full_suite_required=payload.get("full_suite_required"),
            full_suite_reason_codes=tuple(payload.get("full_suite_reason_codes") or ()),
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
        _check_projection(
            payload, field_name="execution_order", actual=result.execution_order
        )
        _check_identity(
            payload,
            result.plan_id,
            names=("plan_id", "content_id"),
            artifact_name="verification plan",
        )
        return result


_RECEIPT_TYPES_BY_SCHEMA: Final[Mapping[str, type[_VerificationContract]]] = (
    MappingProxyType(
        {
            STATIC_ANALYSIS_RECEIPT_SCHEMA: StaticAnalysisReceipt,
            TYPE_CHECK_RECEIPT_SCHEMA: TypeCheckReceipt,
            TEST_RECEIPT_SCHEMA: TestReceipt,
            PROOF_RECEIPT_SCHEMA: ProofReceipt,
        }
    )
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
        raise VerificationContractError(
            "receipts contains more than one result per key"
        )
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
    """Plan-bound receipts plus explicit unresolved requirements."""

    SCHEMA: ClassVar[str] = VERIFICATION_BUNDLE_SCHEMA
    INTERFACE: ClassVar[str] = VERIFICATION_BUNDLE_INTERFACE

    verification_plan: VerificationPlan
    receipts: tuple[VerificationReceipt, ...]
    reused_receipt_cids: tuple[str, ...]
    executed_receipt_cids: tuple[str, ...]
    counterexamples: tuple[CounterexampleReceipt, ...]
    unresolved_requirement_ids: tuple[str, ...]
    human_review_required: bool

    def __post_init__(self) -> None:
        plan = self.verification_plan
        if isinstance(plan, Mapping):
            plan = VerificationPlan.from_dict(plan)
        if not isinstance(plan, VerificationPlan):
            raise VerificationContractError(
                "verification bundle requires a VerificationPlan"
            )
        # Detach all nested key/decision mappings from caller-owned objects.
        plan = VerificationPlan.from_dict(plan.to_record())
        object.__setattr__(self, "verification_plan", plan)
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
        if (
            set(self.reused_receipt_cids) | set(self.executed_receipt_cids)
            != receipt_ids
        ):
            raise VerificationContractError(
                "every bundled receipt must be classified as reused or executed"
            )
        receipts_by_id = {item.receipt_id: item for item in self.receipts}
        if any(
            not receipts_by_id[receipt_cid].terminal_success
            for receipt_cid in self.reused_receipt_cids
        ):
            raise VerificationContractError(
                "reused bundle receipts must be successful terminal evidence"
            )
        decisions_by_key = {
            decision.key_cid: decision
            for decision in self.verification_plan.cache_reuse_decisions
        }
        for receipt_cid in self.reused_receipt_cids:
            receipt = receipts_by_id[receipt_cid]
            decision = decisions_by_key[receipt.key.key_id]
            if (
                decision.disposition is not CacheReuseDisposition.REUSED
                or decision.receipt_cid != receipt_cid
                or decision.candidate_receipt != receipt
            ):
                raise VerificationContractError(
                    "reused bundle receipt is not the exact plan-approved cache hit"
                )
        for receipt_cid in self.executed_receipt_cids:
            receipt = receipts_by_id[receipt_cid]
            decision = decisions_by_key[receipt.key.key_id]
            if decision.disposition is CacheReuseDisposition.REUSED:
                raise VerificationContractError(
                    "executed bundle receipt conflicts with a plan-approved cache hit"
                )
            if decision.candidate_receipt is not None and (
                decision.receipt_cid == receipt_cid
            ):
                raise VerificationContractError(
                    "executed bundle receipt cannot relabel a rejected cache candidate"
                )
        object.__setattr__(
            self, "counterexamples", _counterexamples(self.counterexamples)
        )
        if not {item.failed_receipt_cid for item in self.counterexamples}.issubset(
            receipt_ids
        ):
            raise VerificationIdentityError(
                "counterexample references a receipt outside the bundle"
            )
        if any(
            receipts_by_id[item.failed_receipt_cid].status
            not in {TerminalStatus.FAILED, TerminalStatus.DISPROVED}
            for item in self.counterexamples
        ):
            raise VerificationContractError(
                "counterexample must reference failed or disproved evidence"
            )
        for counterexample in self.counterexamples:
            failed_receipt = receipts_by_id[counterexample.failed_receipt_cid]
            failed_key = failed_receipt.key
            if counterexample.failed_key_cid != failed_key.key_id:
                raise VerificationIdentityError(
                    "counterexample failed key does not match its receipt"
                )
            if counterexample.failed_selector != failed_key.selector_cid:
                raise VerificationIdentityError(
                    "counterexample selector does not match its failed receipt"
                )
            if (
                counterexample.environment_cid != failed_key.environment_cid
                or counterexample.dependency_lock_cid
                != failed_key.dependency_lock_cid
            ):
                raise VerificationIdentityError(
                    "counterexample environment or dependency lock does not match "
                    "its receipt"
                )
            if not set(counterexample.relevant_symbol_version_cids).issubset(
                set(failed_key.affected_symbol_version_cids)
            ):
                raise VerificationIdentityError(
                    "counterexample symbols are outside its failed receipt key"
                )
            if counterexample.reproduction_argv != failed_receipt.execution.command_argv:
                raise VerificationIdentityError(
                    "counterexample reproduction command does not match its failed receipt"
                )
            if failed_key.receipt_kind is VerificationReceiptKind.PROOF:
                if (
                    counterexample.failed_obligation_cid
                    != failed_key.proof_obligation_cid
                ):
                    raise VerificationIdentityError(
                        "proof counterexample obligation does not match its receipt"
                    )
            elif counterexample.failed_obligation_cid:
                raise VerificationIdentityError(
                    "non-proof counterexample cannot name a proof obligation"
                )
        object.__setattr__(
            self,
            "unresolved_requirement_ids",
            _cids(
                self.unresolved_requirement_ids,
                field_name="unresolved_requirement_ids",
            ),
        )
        missing_keys = required - receipt_key_ids
        if missing_keys != set(self.unresolved_requirement_ids):
            raise VerificationContractError(
                "unresolved requirements must equal the missing required receipt keys"
            )
        object.__setattr__(
            self,
            "human_review_required",
            _boolean(self.human_review_required, field_name="human_review_required"),
        )
        if self.verification_plan.human_review_required and not self.human_review_required:
            raise VerificationContractError(
                "bundle cannot downgrade plan-required human review"
            )
        _bounded(
            self,
            artifact_name="verification bundle",
            maximum=2 * MAX_RECORD_BYTES,
        )

    @property
    def plan_cid(self) -> str:
        return self.verification_plan.plan_id

    @property
    def repository_tree_cid(self) -> str:
        return self.verification_plan.repository_tree_cid

    @property
    def environment_cid(self) -> str:
        return self.verification_plan.environment_cid

    @property
    def required_check_key_cids(self) -> tuple[str, ...]:
        return tuple(
            key.key_id for key in self.verification_plan.required_receipt_keys
        )

    @property
    def policy_cid(self) -> str:
        return self.verification_plan.policy_cid

    @property
    def unresolved_obligation_count(self) -> int:
        return len(self.unresolved_proof_obligation_cids)

    @property
    def unresolved_proof_obligation_cids(self) -> tuple[str, ...]:
        receipts_by_key = {receipt.key.key_id: receipt for receipt in self.receipts}
        unresolved: set[str] = set()
        for key in self.verification_plan.required_receipt_keys:
            if key.receipt_kind is not VerificationReceiptKind.PROOF:
                continue
            receipt = receipts_by_key.get(key.key_id)
            if receipt is None or receipt.status not in {
                TerminalStatus.PROVED,
                TerminalStatus.DISPROVED,
            }:
                unresolved.add(key.proof_obligation_cid)
        return tuple(sorted(unresolved))

    @property
    def mandatory_fallback_pending(self) -> bool:
        receipt_key_ids = {receipt.key.key_id for receipt in self.receipts}
        return any(
            key_id not in receipt_key_ids
            for key_id in self.verification_plan.full_suite_receipt_key_cids
        )

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
            "verification_plan": self.verification_plan.to_record(),
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
            "unresolved_proof_obligation_cids": (
                self.unresolved_proof_obligation_cids
            ),
            "unresolved_obligation_count": self.unresolved_obligation_count,
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
            "verification_plan",
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
            "unresolved_proof_obligation_cids",
            "unresolved_obligation_count",
            "structurally_complete",
            "bundle_id",
        }
        _reject_unknown(payload, fields, artifact_name="verification bundle")
        result = cls(
            verification_plan=VerificationPlan.from_dict(
                payload.get("verification_plan") or {}
            ),
            receipts=tuple(payload.get("receipts") or ()),
            reused_receipt_cids=tuple(payload.get("reused_receipt_cids") or ()),
            executed_receipt_cids=tuple(payload.get("executed_receipt_cids") or ()),
            counterexamples=tuple(payload.get("counterexamples") or ()),
            unresolved_requirement_ids=tuple(
                payload.get("unresolved_requirement_ids") or ()
            ),
            human_review_required=payload.get("human_review_required"),
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
        for field_name, actual in (
            ("plan_cid", result.plan_cid),
            ("repository_tree_cid", result.repository_tree_cid),
            ("environment_cid", result.environment_cid),
            ("required_check_key_cids", result.required_check_key_cids),
            ("policy_cid", result.policy_cid),
            (
                "unresolved_proof_obligation_cids",
                result.unresolved_proof_obligation_cids,
            ),
            ("unresolved_obligation_count", result.unresolved_obligation_count),
        ):
            _check_projection(payload, field_name=field_name, actual=actual)
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
            object.__setattr__(self, name, _cids(getattr(self, name), field_name=name))
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
        if (
            self.human_review_required
            != self.model_route_decision.requires_human_review
        ):
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
            dependency_cone_symbols=tuple(payload.get("dependency_cone_symbols") or ()),
            selected_tests=tuple(payload.get("selected_tests") or ()),
            reused_check_key_cids=tuple(payload.get("reused_check_key_cids") or ()),
            executed_check_key_cids=tuple(payload.get("executed_check_key_cids") or ()),
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
    if not normalized:
        return TerminalStatus.UNKNOWN
    for candidate in _FAIL_CLOSED_STATUS_ORDER:
        if candidate in normalized:
            return candidate
    if unresolved:
        return TerminalStatus.UNKNOWN
    if all(item is TerminalStatus.PROVED for item in normalized):
        return TerminalStatus.PROVED
    if all(item.successful for item in normalized):
        return TerminalStatus.PASSED
    return TerminalStatus.UNKNOWN


def _commitment_leaves(
    bundle: VerificationBundle,
) -> tuple[Mapping[str, Any], ...]:
    """Derive Merkle leaves solely from the bundle's typed receipts."""

    leaves = tuple(
        MappingProxyType(
            {
                "key_cid": receipt.key.key_id,
                "receipt_cid": receipt.receipt_id,
                "receipt_kind": receipt.key.receipt_kind.value,
                "status": receipt.status.value,
            }
        )
        for receipt in bundle.receipts
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

    verification_bundle: VerificationBundle

    def __post_init__(self) -> None:
        bundle = self.verification_bundle
        if isinstance(bundle, Mapping):
            bundle = VerificationBundle.from_dict(bundle)
        if not isinstance(bundle, VerificationBundle):
            raise VerificationContractError(
                "verification commitment requires a VerificationBundle"
            )
        # Round-trip through the strict decoder to detach this commitment from
        # every caller-owned receipt and diagnostic mapping.
        bundle = VerificationBundle.from_dict(bundle.to_record())
        object.__setattr__(self, "verification_bundle", bundle)
        _bounded(
            self,
            artifact_name="verification commitment",
            maximum=2 * MAX_RECORD_BYTES,
        )

    @classmethod
    def from_bundle(
        cls,
        verification_bundle: VerificationBundle,
    ) -> VerificationCommitment:
        return cls(verification_bundle=verification_bundle)

    @property
    def repository_tree_cid(self) -> str:
        return self.verification_bundle.repository_tree_cid

    @property
    def environment_cid(self) -> str:
        return self.verification_bundle.environment_cid

    @property
    def required_check_key_cids(self) -> tuple[str, ...]:
        return self.verification_bundle.required_check_key_cids

    @property
    def admitted_leaves(self) -> tuple[Mapping[str, Any], ...]:
        return _commitment_leaves(self.verification_bundle)

    @property
    def unresolved_obligation_count(self) -> int:
        return self.verification_bundle.unresolved_obligation_count

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
        incomplete = int(
            bool(
                self.verification_bundle.mandatory_fallback_pending
                or self.verification_bundle.human_review_required
                or self.verification_bundle.unresolved_requirement_ids
            )
        )
        return aggregate_terminal_status(
            (item["status"] for item in self.admitted_leaves),
            unresolved_obligation_count=incomplete,
        )

    @property
    def public_statement(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "verification-public-statement@1"
                ),
                "verification_bundle_cid": self.verification_bundle.bundle_id,
                "verification_plan_cid": self.verification_bundle.plan_cid,
                "policy_cid": self.verification_bundle.policy_cid,
                "repository_tree_cid": self.repository_tree_cid,
                "environment_cid": self.environment_cid,
                "required_check_set_cid": self.required_check_set_cid,
                "unresolved_obligation_count": self.unresolved_obligation_count,
                "mandatory_fallback_pending": (
                    self.verification_bundle.mandatory_fallback_pending
                ),
                "human_review_required": (
                    self.verification_bundle.human_review_required
                ),
                "aggregate_terminal_status": self.aggregate_terminal_status.value,
            }
        )

    @property
    def commitment_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": VERIFICATION_CONTRACT_VERSION,
            "interface": self.INTERFACE,
            "verification_bundle": self.verification_bundle.to_record(),
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
            "verification_bundle",
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
            verification_bundle=VerificationBundle.from_dict(
                payload.get("verification_bundle") or {}
            )
        )
        for field_name, actual in (
            ("repository_tree_cid", result.repository_tree_cid),
            ("environment_cid", result.environment_cid),
            ("required_check_key_cids", result.required_check_key_cids),
            ("admitted_leaves", result.admitted_leaves),
            ("public_statement", result.public_statement),
            (
                "unresolved_obligation_count",
                result.unresolved_obligation_count,
            ),
        ):
            _check_projection(payload, field_name=field_name, actual=actual)
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


def build_verification_commitment(
    verification_bundle: VerificationBundle,
) -> VerificationCommitment:
    """Build a structural receipt commitment; this is not a ZK proof."""

    return VerificationCommitment.from_bundle(verification_bundle)


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
    "build_verification_commitment",
]

# Public stable spelling for future builders and conformance tests.
TERMINAL_STATUS_PRECEDENCE: Final[tuple[TerminalStatus, ...]] = (
    *_FAIL_CLOSED_STATUS_ORDER,
    TerminalStatus.PROVED,
    TerminalStatus.PASSED,
)
