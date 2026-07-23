"""Deterministic task authority and finite delegation policy.

The supervisor has two deliberately separate authorization lanes:

* :class:`ReferenceAuthorizationEvaluator` is a small, deterministic,
  fail-closed evaluator used for policy enforcement.
* optional Datalog and SecPAL-style adapters check the same finite policy in
  shadow mode.  An executable is reported as supported only after all reviewed
  conformance fixtures agree with the reference evaluator.

Authorization is not program verification.  A permit decision says that a
principal may perform the requested action in the stated lease and policy
context.  It never says that generated code is correct, safe, or complete.
Those claims require their own validation or proof evidence.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
)
from .prover_matrix_registry import (
    CommandRequest,
    CommandResult,
    _default_command_runner as _bounded_command_runner,
)

AUTHORIZATION_LOGIC_VERSION: Final = 1
PRINCIPAL_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/principal@1"
AUTHORIZATION_GRANT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authorization-grant@1"
)
REVOCATION_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/revocation@1"
AUTHORIZATION_POLICY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authorization-policy@1"
)
AUTHORIZATION_REQUEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authorization-request@1"
)
AUTHORIZATION_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authorization-decision@1"
)
AUTHORIZATION_REPORT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authorization-report@1"
)
AUTHORIZATION_ENGINE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authorization-engine-receipt@1"
)
AUTHORIZATION_ENGINE_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authorization-engine-capability@1"
)
AUTHORIZATION_LANE_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authorization-lane-result@1"
)
AUTHORIZATION_FIXTURE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/authorization-fixture@1"
)

DEFAULT_ENGINE_TIMEOUT_SECONDS: Final = 10.0
DEFAULT_MAX_ENGINE_OUTPUT_BYTES: Final = 64 * 1024
DEFAULT_MAX_EXECUTABLE_BYTES: Final = 16 * 1024 * 1024
MAX_DELEGATION_DEPTH: Final = 64
WILDCARD: Final = "*"


class AuthorizationValidationError(ContractValidationError):
    """Raised for malformed or ambiguous authorization contracts."""


class Capability(str, Enum):
    """Supervisor actions that may be granted independently."""

    CLAIM_TASK = "claim_task"
    EXECUTE_TASK = "execute_task"
    PUBLISH_PROGRESS = "publish_progress"
    MERGE = "merge"
    PROMOTE_PROOF = "promote_proof"
    OVERRIDE_POLICY = "override_policy"


AuthorizationCapability = Capability


class AuthorizationVerdict(str, Enum):
    PERMIT = "permit"
    DENY = "deny"


class DenialReason(str, Enum):
    ALLOWED = "allowed"
    NO_APPLICABLE_GRANT = "no_applicable_grant"
    MALFORMED_DELEGATION = "malformed_delegation"
    DELEGATION_DEPTH_EXCEEDED = "delegation_depth_exceeded"
    NOT_YET_VALID = "not_yet_valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    TASK_SCOPE_MISMATCH = "task_scope_mismatch"
    WORKTREE_SCOPE_MISMATCH = "worktree_scope_mismatch"
    PATH_SCOPE_MISMATCH = "path_scope_mismatch"
    LEASE_REQUIRED = "lease_required"
    LEASE_SCOPE_MISMATCH = "lease_scope_mismatch"
    FENCING_EPOCH_REQUIRED = "fencing_epoch_required"
    STALE_FENCING_EPOCH = "stale_fencing_epoch"
    PROOF_AUTHORITY_REQUIRED = "proof_authority_required"
    PROOF_AUTHORITY_MISMATCH = "proof_authority_mismatch"
    OVERRIDE_SCOPE_REQUIRED = "override_scope_required"
    OVERRIDE_SCOPE_MISMATCH = "override_scope_mismatch"


class GeneratedCodeCorrectness(str, Enum):
    """The only correctness projection authorization evidence may carry."""

    NOT_ESTABLISHED = "not_established"


class AuthorizationEngine(str, Enum):
    DATALOG = "datalog"
    SECPAL = "secpal"


PolicyEngine = AuthorizationEngine


class EngineSupportStatus(str, Enum):
    UNSUPPORTED = "unsupported"
    CONFORMANT = "conformant"


class ConformanceStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    ERROR = "error"
    NOT_RUN = "not_run"


class LaneStatus(str, Enum):
    AGREED = "agreed"
    DISAGREED = "disagreed"
    UNSUPPORTED = "unsupported"
    ERROR = "error"


LEASE_BOUND_CAPABILITIES: Final = frozenset(
    {
        Capability.EXECUTE_TASK,
        Capability.PUBLISH_PROGRESS,
        Capability.MERGE,
        Capability.PROMOTE_PROOF,
        Capability.OVERRIDE_POLICY,
    }
)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise AuthorizationValidationError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise AuthorizationValidationError(f"{name} must not be empty")
    if "\x00" in result:
        raise AuthorizationValidationError(f"{name} must not contain NUL")
    return result


def _strings(
    values: Iterable[Any] | Any,
    name: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if values is None:
        items: Iterable[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Iterable) and not isinstance(
        values, (bytes, bytearray, Mapping)
    ):
        items = values
    else:
        raise AuthorizationValidationError(f"{name} must be a sequence of strings")
    result = tuple(sorted({_text(item, name) for item in items}))
    if required and not result:
        raise AuthorizationValidationError(f"{name} must not be empty")
    return result


def _ordered_strings(
    values: Iterable[Any] | Any,
    name: str,
    *,
    required: bool = False,
) -> tuple[str, ...]:
    if values is None:
        items: Iterable[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Iterable) and not isinstance(
        values, (bytes, bytearray, Mapping)
    ):
        items = values
    else:
        raise AuthorizationValidationError(f"{name} must be a sequence of strings")
    result: list[str] = []
    for item in items:
        normalized = _text(item, name)
        if normalized in result:
            raise AuthorizationValidationError(f"{name} must not contain duplicates")
        result.append(normalized)
    if required and not result:
        raise AuthorizationValidationError(f"{name} must not be empty")
    return tuple(result)


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuthorizationValidationError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative(value: Any, name: str) -> int | None:
    return None if value is None else _nonnegative(value, name)


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise AuthorizationValidationError(f"unsupported {name}: {value!r}") from exc


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise AuthorizationValidationError(
            f"unsupported schema {supplied!r}; expected {expected}"
        )


def _identity(payload: Mapping[str, Any], actual: str, noun: str) -> None:
    claimed = payload.get("content_id") or payload.get("identity")
    if claimed and claimed != actual:
        raise AuthorizationValidationError(f"{noun} identity does not match payload")


def _normalized_path(value: str) -> str:
    result = _text(value, "path", required=False).replace("\\", "/")
    while "//" in result:
        result = result.replace("//", "/")
    if result.startswith("/"):
        result = result[1:]
    components = tuple(item for item in result.split("/") if item not in ("", "."))
    if any(item == ".." for item in components):
        raise AuthorizationValidationError("path scope must not traverse a parent")
    return "/".join(components)


def _normalize_scopes(
    values: Iterable[Any] | Any,
    name: str,
    *,
    paths: bool = False,
    required: bool = False,
) -> tuple[str, ...]:
    result = _strings(values, name, required=required)
    if WILDCARD in result and len(result) != 1:
        raise AuthorizationValidationError(f"{name} wildcard must stand alone")
    if paths:
        normalized = tuple(
            sorted(
                {
                    WILDCARD if item == WILDCARD else _normalized_path(item)
                    for item in result
                }
            )
        )
        if any(not item for item in normalized):
            raise AuthorizationValidationError(f"{name} must not contain an empty path")
        return normalized
    return result


def _scope_contains(scope: Sequence[str], value: str) -> bool:
    return WILDCARD in scope or value in scope


def _path_contains(prefixes: Sequence[str], path: str) -> bool:
    normalized = _normalized_path(path)
    return any(
        prefix == WILDCARD
        or normalized == prefix
        or normalized.startswith(prefix.rstrip("/") + "/")
        for prefix in prefixes
    )


def _item_covered(child: str, parent: str, *, paths: bool) -> bool:
    if parent == WILDCARD:
        return True
    if child == WILDCARD:
        return False
    if paths:
        return child == parent or child.startswith(parent.rstrip("/") + "/")
    return child == parent


def _scope_subset(
    child: Sequence[str], parent: Sequence[str], *, paths: bool = False
) -> bool:
    return all(
        any(_item_covered(item, outer, paths=paths) for outer in parent)
        for item in child
    )


@dataclass(frozen=True)
class Principal(CanonicalContract):
    """A stable policy principal, independent of display names."""

    SCHEMA = PRINCIPAL_SCHEMA

    principal_id: str
    kind: str = "agent"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "principal_id", _text(self.principal_id, "principal_id")
        )
        object.__setattr__(self, "kind", _text(self.kind, "kind"))

    def _payload(self) -> dict[str, Any]:
        return {
            "authorization_logic_version": AUTHORIZATION_LOGIC_VERSION,
            "principal_id": self.principal_id,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Principal":
        if not isinstance(payload, Mapping):
            raise AuthorizationValidationError("principal must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            principal_id=payload.get("principal_id", ""),
            kind=payload.get("kind", "agent"),
        )
        _identity(payload, result.content_id, "principal")
        return result


def _principal_id(value: Principal | str, name: str) -> str:
    return value.principal_id if isinstance(value, Principal) else _text(value, name)


@dataclass(frozen=True)
class AuthorizationGrant(CanonicalContract):
    """A SecPAL-style ``issuer says subject may capability`` assertion.

    ``delegation_depth`` is the number of additional delegation hops the
    subject may make.  A child grant must name its parent and use a strictly
    smaller depth, which prevents ambient-authority and confused-deputy chains.
    """

    SCHEMA = AUTHORIZATION_GRANT_SCHEMA

    statement_id: str
    issuer: Principal | str
    subject: Principal | str
    capability: Capability
    task_scope: tuple[str, ...]
    delegation_depth: int = 0
    parent_statement_id: str | None = None
    lease_scope: tuple[str, ...] = (WILDCARD,)
    worktree_scope: tuple[str, ...] = (WILDCARD,)
    path_scope: tuple[str, ...] = (WILDCARD,)
    fencing_epoch: int | None = None
    proof_authorities: tuple[str, ...] = ()
    override_scopes: tuple[str, ...] = ()
    not_before_ms: int = 0
    expires_at_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "statement_id", _text(self.statement_id, "statement_id")
        )
        object.__setattr__(self, "issuer", _principal_id(self.issuer, "issuer"))
        object.__setattr__(self, "subject", _principal_id(self.subject, "subject"))
        object.__setattr__(
            self, "capability", _enum(self.capability, Capability, "capability")
        )
        object.__setattr__(
            self,
            "task_scope",
            _normalize_scopes(self.task_scope, "task_scope", required=True),
        )
        depth = _nonnegative(self.delegation_depth, "delegation_depth")
        if depth > MAX_DELEGATION_DEPTH:
            raise AuthorizationValidationError(
                f"delegation_depth exceeds {MAX_DELEGATION_DEPTH}"
            )
        object.__setattr__(self, "delegation_depth", depth)
        parent = self.parent_statement_id
        object.__setattr__(
            self,
            "parent_statement_id",
            None if parent in (None, "") else _text(parent, "parent_statement_id"),
        )
        object.__setattr__(
            self,
            "lease_scope",
            _normalize_scopes(self.lease_scope, "lease_scope", required=True),
        )
        object.__setattr__(
            self,
            "worktree_scope",
            _normalize_scopes(self.worktree_scope, "worktree_scope", required=True),
        )
        object.__setattr__(
            self,
            "path_scope",
            _normalize_scopes(self.path_scope, "path_scope", paths=True, required=True),
        )
        object.__setattr__(
            self,
            "fencing_epoch",
            _optional_nonnegative(self.fencing_epoch, "fencing_epoch"),
        )
        object.__setattr__(
            self,
            "proof_authorities",
            _normalize_scopes(self.proof_authorities, "proof_authorities"),
        )
        object.__setattr__(
            self,
            "override_scopes",
            _normalize_scopes(self.override_scopes, "override_scopes", paths=True),
        )
        not_before = _nonnegative(self.not_before_ms, "not_before_ms")
        expires = _optional_nonnegative(self.expires_at_ms, "expires_at_ms")
        if expires is not None and expires <= not_before:
            raise AuthorizationValidationError(
                "expires_at_ms must be greater than not_before_ms"
            )
        object.__setattr__(self, "not_before_ms", not_before)
        object.__setattr__(self, "expires_at_ms", expires)

    @property
    def can_delegate(self) -> bool:
        return self.delegation_depth > 0

    def _payload(self) -> dict[str, Any]:
        return {
            "authorization_logic_version": AUTHORIZATION_LOGIC_VERSION,
            "statement_id": self.statement_id,
            "issuer": self.issuer,
            "subject": self.subject,
            "capability": self.capability,
            "task_scope": self.task_scope,
            "delegation_depth": self.delegation_depth,
            "parent_statement_id": self.parent_statement_id,
            "lease_scope": self.lease_scope,
            "worktree_scope": self.worktree_scope,
            "path_scope": self.path_scope,
            "fencing_epoch": self.fencing_epoch,
            "proof_authorities": self.proof_authorities,
            "override_scopes": self.override_scopes,
            "not_before_ms": self.not_before_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizationGrant":
        if not isinstance(payload, Mapping):
            raise AuthorizationValidationError("authorization grant must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            statement_id=payload.get("statement_id", ""),
            issuer=payload.get("issuer", ""),
            subject=payload.get("subject", ""),
            capability=payload.get("capability", ""),
            task_scope=tuple(payload.get("task_scope") or ()),
            delegation_depth=payload.get("delegation_depth", 0),
            parent_statement_id=payload.get("parent_statement_id"),
            lease_scope=tuple(payload.get("lease_scope") or ()),
            worktree_scope=tuple(payload.get("worktree_scope") or ()),
            path_scope=tuple(payload.get("path_scope") or ()),
            fencing_epoch=payload.get("fencing_epoch"),
            proof_authorities=tuple(payload.get("proof_authorities") or ()),
            override_scopes=tuple(payload.get("override_scopes") or ()),
            not_before_ms=payload.get("not_before_ms", 0),
            expires_at_ms=payload.get("expires_at_ms"),
        )
        _identity(payload, result.content_id, "authorization grant")
        return result


PolicyGrant = AuthorizationGrant
DelegationGrant = AuthorizationGrant


@dataclass(frozen=True)
class Revocation(CanonicalContract):
    """An issuer- or root-authorized, time-bound statement revocation."""

    SCHEMA = REVOCATION_SCHEMA

    revocation_id: str
    issuer: Principal | str
    target_statement_id: str
    effective_at_ms: int
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "revocation_id", _text(self.revocation_id, "revocation_id")
        )
        object.__setattr__(self, "issuer", _principal_id(self.issuer, "issuer"))
        object.__setattr__(
            self,
            "target_statement_id",
            _text(self.target_statement_id, "target_statement_id"),
        )
        object.__setattr__(
            self,
            "effective_at_ms",
            _nonnegative(self.effective_at_ms, "effective_at_ms"),
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))

    def _payload(self) -> dict[str, Any]:
        return {
            "authorization_logic_version": AUTHORIZATION_LOGIC_VERSION,
            "revocation_id": self.revocation_id,
            "issuer": self.issuer,
            "target_statement_id": self.target_statement_id,
            "effective_at_ms": self.effective_at_ms,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Revocation":
        if not isinstance(payload, Mapping):
            raise AuthorizationValidationError("revocation must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            revocation_id=payload.get("revocation_id", ""),
            issuer=payload.get("issuer", ""),
            target_statement_id=payload.get("target_statement_id", ""),
            effective_at_ms=payload.get("effective_at_ms", -1),
            reason=payload.get("reason", ""),
        )
        _identity(payload, result.content_id, "revocation")
        return result


@dataclass(frozen=True)
class AuthorizationPolicy(CanonicalContract):
    """Finite authority graph plus authoritative lease/fence state."""

    SCHEMA = AUTHORIZATION_POLICY_SCHEMA

    policy_id: str
    version: str
    trusted_roots: tuple[str, ...]
    grants: tuple[AuthorizationGrant, ...]
    revocations: tuple[Revocation, ...] = ()
    current_fencing_epochs: Mapping[str, int] = field(default_factory=dict)
    current_lease_ids: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, "policy_id"))
        object.__setattr__(self, "version", _text(self.version, "version"))
        roots = _strings(self.trusted_roots, "trusted_roots", required=True)
        object.__setattr__(self, "trusted_roots", roots)
        grants = tuple(self.grants)
        if any(not isinstance(item, AuthorizationGrant) for item in grants):
            raise AuthorizationValidationError(
                "grants must contain AuthorizationGrant values"
            )
        grants = tuple(sorted(grants, key=lambda item: item.statement_id))
        grant_ids = [item.statement_id for item in grants]
        if len(grant_ids) != len(set(grant_ids)):
            raise AuthorizationValidationError(
                "grant statement_id values must be unique"
            )
        object.__setattr__(self, "grants", grants)
        revocations = tuple(self.revocations)
        if any(not isinstance(item, Revocation) for item in revocations):
            raise AuthorizationValidationError(
                "revocations must contain Revocation values"
            )
        revocations = tuple(sorted(revocations, key=lambda item: item.revocation_id))
        revocation_ids = [item.revocation_id for item in revocations]
        if len(revocation_ids) != len(set(revocation_ids)):
            raise AuthorizationValidationError("revocation_id values must be unique")
        unknown_targets = {item.target_statement_id for item in revocations} - set(
            grant_ids
        )
        if unknown_targets:
            raise AuthorizationValidationError(
                "revocation targets unknown grant statements"
            )
        object.__setattr__(self, "revocations", revocations)

        epochs: dict[str, int] = {}
        if not isinstance(self.current_fencing_epochs, Mapping):
            raise AuthorizationValidationError(
                "current_fencing_epochs must be an object"
            )
        for raw_task, raw_epoch in self.current_fencing_epochs.items():
            task = _text(raw_task, "current_fencing_epochs key")
            epochs[task] = _nonnegative(raw_epoch, "current_fencing_epochs value")
        object.__setattr__(self, "current_fencing_epochs", dict(sorted(epochs.items())))

        leases: dict[str, str] = {}
        if not isinstance(self.current_lease_ids, Mapping):
            raise AuthorizationValidationError("current_lease_ids must be an object")
        for raw_task, raw_lease in self.current_lease_ids.items():
            task = _text(raw_task, "current_lease_ids key")
            leases[task] = _text(raw_lease, "current_lease_ids value")
        object.__setattr__(self, "current_lease_ids", dict(sorted(leases.items())))

    @property
    def grant_index(self) -> Mapping[str, AuthorizationGrant]:
        return {item.statement_id: item for item in self.grants}

    def _payload(self) -> dict[str, Any]:
        return {
            "authorization_logic_version": AUTHORIZATION_LOGIC_VERSION,
            "policy_id": self.policy_id,
            "version": self.version,
            "trusted_roots": self.trusted_roots,
            "grants": self.grants,
            "revocations": self.revocations,
            "current_fencing_epochs": self.current_fencing_epochs,
            "current_lease_ids": self.current_lease_ids,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizationPolicy":
        if not isinstance(payload, Mapping):
            raise AuthorizationValidationError("authorization policy must be an object")
        _schema(payload, cls.SCHEMA)
        raw_grants = payload.get("grants") or ()
        raw_revocations = payload.get("revocations") or ()
        result = cls(
            policy_id=payload.get("policy_id", ""),
            version=payload.get("version", ""),
            trusted_roots=tuple(payload.get("trusted_roots") or ()),
            grants=tuple(
                (
                    item
                    if isinstance(item, AuthorizationGrant)
                    else AuthorizationGrant.from_dict(item)
                )
                for item in raw_grants
            ),
            revocations=tuple(
                item if isinstance(item, Revocation) else Revocation.from_dict(item)
                for item in raw_revocations
            ),
            current_fencing_epochs=payload.get("current_fencing_epochs") or {},
            current_lease_ids=payload.get("current_lease_ids") or {},
        )
        _identity(payload, result.content_id, "authorization policy")
        return result


AuthorityPolicy = AuthorizationPolicy


@dataclass(frozen=True)
class AuthorizationRequest(CanonicalContract):
    """One fully scoped action authorization query."""

    SCHEMA = AUTHORIZATION_REQUEST_SCHEMA

    principal: Principal | str
    capability: Capability
    task_id: str
    evaluated_at_ms: int
    lease_id: str | None = None
    worktree_id: str | None = None
    path: str | None = None
    fencing_epoch: int | None = None
    proof_authority: str | None = None
    override_scope: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "principal", _principal_id(self.principal, "principal")
        )
        capability = _enum(self.capability, Capability, "capability")
        object.__setattr__(self, "capability", capability)
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(
            self,
            "evaluated_at_ms",
            _nonnegative(self.evaluated_at_ms, "evaluated_at_ms"),
        )
        for name in ("lease_id", "worktree_id", "proof_authority"):
            value = getattr(self, name)
            object.__setattr__(
                self, name, None if value in (None, "") else _text(value, name)
            )
        object.__setattr__(
            self,
            "path",
            None if self.path in (None, "") else _normalized_path(str(self.path)),
        )
        object.__setattr__(
            self,
            "override_scope",
            (
                None
                if self.override_scope in (None, "")
                else _normalized_path(str(self.override_scope))
            ),
        )
        object.__setattr__(
            self,
            "fencing_epoch",
            _optional_nonnegative(self.fencing_epoch, "fencing_epoch"),
        )

    @property
    def actor(self) -> str:
        return str(self.principal)

    def _payload(self) -> dict[str, Any]:
        return {
            "authorization_logic_version": AUTHORIZATION_LOGIC_VERSION,
            "principal": self.principal,
            "capability": self.capability,
            "task_id": self.task_id,
            "evaluated_at_ms": self.evaluated_at_ms,
            "lease_id": self.lease_id,
            "worktree_id": self.worktree_id,
            "path": self.path,
            "fencing_epoch": self.fencing_epoch,
            "proof_authority": self.proof_authority,
            "override_scope": self.override_scope,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizationRequest":
        if not isinstance(payload, Mapping):
            raise AuthorizationValidationError(
                "authorization request must be an object"
            )
        _schema(payload, cls.SCHEMA)
        result = cls(
            principal=payload.get("principal", payload.get("actor", "")),
            capability=payload.get("capability", ""),
            task_id=payload.get("task_id", ""),
            evaluated_at_ms=payload.get("evaluated_at_ms", -1),
            lease_id=payload.get("lease_id"),
            worktree_id=payload.get("worktree_id"),
            path=payload.get("path"),
            fencing_epoch=payload.get("fencing_epoch"),
            proof_authority=payload.get("proof_authority"),
            override_scope=payload.get("override_scope"),
        )
        _identity(payload, result.content_id, "authorization request")
        return result


ActionRequest = AuthorizationRequest


@dataclass(frozen=True)
class AuthorizationDecision(CanonicalContract):
    """Reference decision and its minimal authority chain."""

    SCHEMA = AUTHORIZATION_DECISION_SCHEMA

    verdict: AuthorizationVerdict
    reason: DenialReason
    policy_identity: str
    request_identity: str
    evaluated_at_ms: int
    matched_statement_ids: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        verdict = _enum(self.verdict, AuthorizationVerdict, "authorization verdict")
        reason = _enum(self.reason, DenialReason, "denial reason")
        chain = _ordered_strings(
            self.matched_statement_ids,
            "matched_statement_ids",
        )
        if verdict is AuthorizationVerdict.PERMIT:
            if reason is not DenialReason.ALLOWED or not chain:
                raise AuthorizationValidationError(
                    "permit decision requires an allowed reason and authority chain"
                )
        elif reason is DenialReason.ALLOWED or chain:
            raise AuthorizationValidationError(
                "deny decision cannot carry an allowed chain"
            )
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self, "policy_identity", _text(self.policy_identity, "policy_identity")
        )
        object.__setattr__(
            self, "request_identity", _text(self.request_identity, "request_identity")
        )
        object.__setattr__(
            self,
            "evaluated_at_ms",
            _nonnegative(self.evaluated_at_ms, "evaluated_at_ms"),
        )
        object.__setattr__(self, "matched_statement_ids", chain)
        object.__setattr__(self, "detail", _text(self.detail, "detail", required=False))

    @property
    def permitted(self) -> bool:
        return self.verdict is AuthorizationVerdict.PERMIT

    @property
    def permits_action(self) -> bool:
        return self.permitted

    @property
    def establishes_generated_code_correctness(self) -> bool:
        return False

    @property
    def generated_code_correctness(self) -> GeneratedCodeCorrectness:
        return GeneratedCodeCorrectness.NOT_ESTABLISHED

    def _payload(self) -> dict[str, Any]:
        return {
            "authorization_logic_version": AUTHORIZATION_LOGIC_VERSION,
            "verdict": self.verdict,
            "reason": self.reason,
            "policy_identity": self.policy_identity,
            "request_identity": self.request_identity,
            "evaluated_at_ms": self.evaluated_at_ms,
            "matched_statement_ids": self.matched_statement_ids,
            "detail": self.detail,
            "permits_action": self.permitted,
            "establishes_generated_code_correctness": False,
            "generated_code_correctness": self.generated_code_correctness,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizationDecision":
        if not isinstance(payload, Mapping):
            raise AuthorizationValidationError(
                "authorization decision must be an object"
            )
        _schema(payload, cls.SCHEMA)
        if payload.get("establishes_generated_code_correctness") not in (None, False):
            raise AuthorizationValidationError(
                "authorization cannot establish generated-code correctness"
            )
        if payload.get("generated_code_correctness") not in (
            None,
            "",
            GeneratedCodeCorrectness.NOT_ESTABLISHED.value,
        ):
            raise AuthorizationValidationError(
                "authorization cannot establish generated-code correctness"
            )
        result = cls(
            verdict=payload.get("verdict", ""),
            reason=payload.get("reason", ""),
            policy_identity=payload.get("policy_identity", ""),
            request_identity=payload.get("request_identity", ""),
            evaluated_at_ms=payload.get("evaluated_at_ms", -1),
            matched_statement_ids=tuple(payload.get("matched_statement_ids") or ()),
            detail=payload.get("detail", ""),
        )
        claimed_permit = payload.get("permits_action")
        if claimed_permit is not None and claimed_permit is not result.permitted:
            raise AuthorizationValidationError(
                "permit projection does not match verdict"
            )
        _identity(payload, result.content_id, "authorization decision")
        return result


AuthorizationEvidence = AuthorizationDecision


_REASON_PRIORITY: Final = (
    DenialReason.REVOKED,
    DenialReason.STALE_FENCING_EPOCH,
    DenialReason.LEASE_SCOPE_MISMATCH,
    DenialReason.LEASE_REQUIRED,
    DenialReason.FENCING_EPOCH_REQUIRED,
    DenialReason.EXPIRED,
    DenialReason.NOT_YET_VALID,
    DenialReason.DELEGATION_DEPTH_EXCEEDED,
    DenialReason.MALFORMED_DELEGATION,
    DenialReason.PROOF_AUTHORITY_REQUIRED,
    DenialReason.PROOF_AUTHORITY_MISMATCH,
    DenialReason.OVERRIDE_SCOPE_REQUIRED,
    DenialReason.OVERRIDE_SCOPE_MISMATCH,
    DenialReason.WORKTREE_SCOPE_MISMATCH,
    DenialReason.PATH_SCOPE_MISMATCH,
    DenialReason.TASK_SCOPE_MISMATCH,
    DenialReason.NO_APPLICABLE_GRANT,
)


class ReferenceAuthorizationEvaluator:
    """Evaluate a finite parent-linked delegation graph without ambient rights."""

    def evaluate(
        self, policy: AuthorizationPolicy, request: AuthorizationRequest
    ) -> AuthorizationDecision:
        if not isinstance(policy, AuthorizationPolicy):
            raise AuthorizationValidationError("policy must be an AuthorizationPolicy")
        if not isinstance(request, AuthorizationRequest):
            raise AuthorizationValidationError(
                "request must be an AuthorizationRequest"
            )

        candidates = tuple(
            grant
            for grant in policy.grants
            if grant.subject == request.principal
            and grant.capability is request.capability
        )
        failures: list[DenialReason] = []
        for leaf in candidates:
            scope_failure = self._request_failure(policy, request, leaf)
            if scope_failure is not None:
                failures.append(scope_failure)
                continue
            chain, chain_failure = self._authority_chain(policy, request, leaf)
            if chain_failure is None:
                return AuthorizationDecision(
                    verdict=AuthorizationVerdict.PERMIT,
                    reason=DenialReason.ALLOWED,
                    policy_identity=policy.content_id,
                    request_identity=request.content_id,
                    evaluated_at_ms=request.evaluated_at_ms,
                    matched_statement_ids=tuple(item.statement_id for item in chain),
                    detail="a current, scope-preserving delegation chain authorizes the action",
                )
            failures.append(chain_failure)

        reason = next(
            (item for item in _REASON_PRIORITY if item in failures),
            DenialReason.NO_APPLICABLE_GRANT,
        )
        return AuthorizationDecision(
            verdict=AuthorizationVerdict.DENY,
            reason=reason,
            policy_identity=policy.content_id,
            request_identity=request.content_id,
            evaluated_at_ms=request.evaluated_at_ms,
            detail=self._detail(reason),
        )

    authorize = evaluate
    check = evaluate

    @staticmethod
    def _detail(reason: DenialReason) -> str:
        return {
            DenialReason.NO_APPLICABLE_GRANT: (
                "no grant names this principal and exact capability"
            ),
            DenialReason.REVOKED: "the authority chain contains an effective revocation",
            DenialReason.STALE_FENCING_EPOCH: (
                "the request or grant does not match the current fencing epoch"
            ),
            DenialReason.LEASE_REQUIRED: "the capability requires an explicit lease",
            DenialReason.FENCING_EPOCH_REQUIRED: (
                "the capability requires an explicit fencing epoch"
            ),
        }.get(reason, reason.value.replace("_", " "))

    @staticmethod
    def _request_failure(
        policy: AuthorizationPolicy,
        request: AuthorizationRequest,
        grant: AuthorizationGrant,
    ) -> DenialReason | None:
        if not _scope_contains(grant.task_scope, request.task_id):
            return DenialReason.TASK_SCOPE_MISMATCH
        if request.worktree_id is not None and not _scope_contains(
            grant.worktree_scope, request.worktree_id
        ):
            return DenialReason.WORKTREE_SCOPE_MISMATCH
        if request.path is not None and not _path_contains(
            grant.path_scope, request.path
        ):
            return DenialReason.PATH_SCOPE_MISMATCH
        if request.evaluated_at_ms < grant.not_before_ms:
            return DenialReason.NOT_YET_VALID
        if (
            grant.expires_at_ms is not None
            and request.evaluated_at_ms >= grant.expires_at_ms
        ):
            return DenialReason.EXPIRED

        if request.capability in LEASE_BOUND_CAPABILITIES:
            if request.lease_id is None:
                return DenialReason.LEASE_REQUIRED
            if request.fencing_epoch is None:
                return DenialReason.FENCING_EPOCH_REQUIRED
            if not _scope_contains(grant.lease_scope, request.lease_id):
                return DenialReason.LEASE_SCOPE_MISMATCH
            current_lease = policy.current_lease_ids.get(request.task_id)
            if current_lease is None or request.lease_id != current_lease:
                return DenialReason.LEASE_SCOPE_MISMATCH
            current_epoch = policy.current_fencing_epochs.get(request.task_id)
            if (
                current_epoch is None
                or request.fencing_epoch != current_epoch
                or grant.fencing_epoch != current_epoch
            ):
                return DenialReason.STALE_FENCING_EPOCH
        elif grant.fencing_epoch is not None:
            current_epoch = policy.current_fencing_epochs.get(request.task_id)
            if request.fencing_epoch != grant.fencing_epoch or (
                current_epoch is not None and request.fencing_epoch != current_epoch
            ):
                return DenialReason.STALE_FENCING_EPOCH

        if request.capability is Capability.PROMOTE_PROOF:
            if request.proof_authority is None:
                return DenialReason.PROOF_AUTHORITY_REQUIRED
            if not _scope_contains(grant.proof_authorities, request.proof_authority):
                return DenialReason.PROOF_AUTHORITY_MISMATCH
        elif request.proof_authority is not None:
            # Proof claims cannot be smuggled into an unrelated capability.
            return DenialReason.PROOF_AUTHORITY_MISMATCH

        if request.capability is Capability.OVERRIDE_POLICY:
            if request.override_scope is None:
                return DenialReason.OVERRIDE_SCOPE_REQUIRED
            if not _path_contains(grant.override_scopes, request.override_scope):
                return DenialReason.OVERRIDE_SCOPE_MISMATCH
        elif request.override_scope is not None:
            return DenialReason.OVERRIDE_SCOPE_MISMATCH
        return None

    @staticmethod
    def _authorized_revocations(
        policy: AuthorizationPolicy,
        now_ms: int,
    ) -> frozenset[str]:
        index = policy.grant_index
        result: set[str] = set()
        for revocation in policy.revocations:
            if revocation.effective_at_ms > now_ms:
                continue
            target = index[revocation.target_statement_id]
            if (
                revocation.issuer in policy.trusted_roots
                or revocation.issuer == target.issuer
            ):
                result.add(target.statement_id)
        return frozenset(result)

    def _authority_chain(
        self,
        policy: AuthorizationPolicy,
        request: AuthorizationRequest,
        leaf: AuthorizationGrant,
    ) -> tuple[tuple[AuthorizationGrant, ...], DenialReason | None]:
        index = policy.grant_index
        revoked = self._authorized_revocations(policy, request.evaluated_at_ms)
        reverse_chain: list[AuthorizationGrant] = []
        seen: set[str] = set()
        current = leaf
        while True:
            if (
                current.statement_id in seen
                or len(reverse_chain) > MAX_DELEGATION_DEPTH
            ):
                return (), DenialReason.MALFORMED_DELEGATION
            seen.add(current.statement_id)
            reverse_chain.append(current)
            if current.statement_id in revoked:
                return (), DenialReason.REVOKED
            if request.evaluated_at_ms < current.not_before_ms:
                return (), DenialReason.NOT_YET_VALID
            if (
                current.expires_at_ms is not None
                and request.evaluated_at_ms >= current.expires_at_ms
            ):
                return (), DenialReason.EXPIRED
            # Every link, including a root assertion, is scoped to the exact
            # request.  Checking only the leaf would let a current child grant
            # launder a stale parent lease or fencing epoch.
            request_failure = self._request_failure(policy, request, current)
            if request_failure is not None:
                return (), request_failure
            parent_id = current.parent_statement_id
            if parent_id is None:
                if current.issuer not in policy.trusted_roots:
                    return (), DenialReason.MALFORMED_DELEGATION
                return tuple(reversed(reverse_chain)), None
            parent = index.get(parent_id)
            if parent is None:
                return (), DenialReason.MALFORMED_DELEGATION
            if (
                parent.subject != current.issuer
                or parent.capability is not current.capability
            ):
                return (), DenialReason.MALFORMED_DELEGATION
            if parent.delegation_depth < 1 or (
                current.delegation_depth >= parent.delegation_depth
            ):
                return (), DenialReason.DELEGATION_DEPTH_EXCEEDED
            if not self._narrows(current, parent):
                return (), DenialReason.MALFORMED_DELEGATION
            current = parent

    @staticmethod
    def _narrows(child: AuthorizationGrant, parent: AuthorizationGrant) -> bool:
        return (
            _scope_subset(child.task_scope, parent.task_scope)
            and _scope_subset(child.lease_scope, parent.lease_scope)
            and _scope_subset(child.worktree_scope, parent.worktree_scope)
            and _scope_subset(child.path_scope, parent.path_scope, paths=True)
            and _scope_subset(child.proof_authorities, parent.proof_authorities)
            and _scope_subset(child.override_scopes, parent.override_scopes, paths=True)
            and (
                parent.fencing_epoch is None
                or child.fencing_epoch == parent.fencing_epoch
            )
            and child.not_before_ms >= parent.not_before_ms
            and (
                parent.expires_at_ms is None
                or (
                    child.expires_at_ms is not None
                    and child.expires_at_ms <= parent.expires_at_ms
                )
            )
        )


PolicyEvaluator = ReferenceAuthorizationEvaluator
AuthorizationEvaluator = ReferenceAuthorizationEvaluator


def evaluate_authorization(
    policy: AuthorizationPolicy, request: AuthorizationRequest
) -> AuthorizationDecision:
    """Evaluate one request with the deterministic reference semantics."""

    return ReferenceAuthorizationEvaluator().evaluate(policy, request)


authorize = evaluate_authorization


def _logic_atom(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def render_datalog_policy(
    policy: AuthorizationPolicy, request: AuthorizationRequest
) -> str:
    """Render a deterministic finite Soufflé program for the same query.

    Scope matching and narrowing are finite translator facts; authority
    chaining, parent identity, depth, validity, and revocation are checked by
    recursive Datalog rules.
    """

    evaluator = ReferenceAuthorizationEvaluator()
    revoked = evaluator._authorized_revocations(policy, request.evaluated_at_ms)
    lines = [
        "// supervisor authorization policy; generated deterministically",
        ".decl grant(id:symbol, issuer:symbol, subject:symbol, capability:symbol, depth:number, parent:symbol)",
        ".decl trusted(principal:symbol)",
        ".decl active(id:symbol)",
        ".decl revoked(id:symbol)",
        ".decl narrows(child:symbol, parent:symbol)",
        ".decl request_match(id:symbol)",
        ".decl authorized(id:symbol)",
        ".decl authz_result(verdict:symbol)",
        ".output authz_result(IO=stdout)",
    ]
    for root in policy.trusted_roots:
        lines.append(f"trusted({_logic_atom(root)}).")
    for grant in policy.grants:
        parent = grant.parent_statement_id or ""
        lines.append(
            "grant(%s,%s,%s,%s,%d,%s)."
            % (
                _logic_atom(grant.statement_id),
                _logic_atom(str(grant.issuer)),
                _logic_atom(str(grant.subject)),
                _logic_atom(grant.capability.value),
                grant.delegation_depth,
                _logic_atom(parent),
            )
        )
        if request.evaluated_at_ms >= grant.not_before_ms and (
            grant.expires_at_ms is None or request.evaluated_at_ms < grant.expires_at_ms
        ):
            lines.append(f"active({_logic_atom(grant.statement_id)}).")
        if grant.statement_id in revoked:
            lines.append(f"revoked({_logic_atom(grant.statement_id)}).")
        if evaluator._request_failure(policy, request, grant) is None:
            lines.append(f"request_match({_logic_atom(grant.statement_id)}).")
        if grant.parent_statement_id:
            parent_grant = policy.grant_index.get(grant.parent_statement_id)
            if parent_grant is not None and evaluator._narrows(grant, parent_grant):
                lines.append(
                    f"narrows({_logic_atom(grant.statement_id)},"
                    f"{_logic_atom(parent_grant.statement_id)})."
                )
    lines.extend(
        [
            'authorized(G) :- grant(G,I,_,_,_,""), trusted(I), active(G), '
            "!revoked(G), request_match(G).",
            'authorized(G) :- grant(G,I,_,C,D,P), P != "", grant(P,_,I,C,PD,_), '
            "PD > D, authorized(P), active(G), !revoked(G), narrows(G,P), "
            "request_match(G).",
            'authz_result("PERMIT") :- grant(G,_,'
            f"{_logic_atom(str(request.principal))},"
            f"{_logic_atom(request.capability.value)},_,_), "
            "authorized(G), request_match(G).",
        ]
    )
    return "\n".join(lines) + "\n"


render_datalog = render_datalog_policy


def render_secpal_policy(
    policy: AuthorizationPolicy, request: AuthorizationRequest
) -> str:
    """Render a canonical, human-auditable SecPAL-style assertion document."""

    lines = [
        "# supervisor SecPAL-style authorization policy v1",
        "# assertions: issuer says subject can action; issuer revokes statement",
        f"# policy {policy.content_id}",
    ]
    for root in policy.trusted_roots:
        lines.append(f'trust "{root}".')
    for grant in policy.grants:
        parent = (
            f' under "{grant.parent_statement_id}"' if grant.parent_statement_id else ""
        )
        lines.append(
            f'"{grant.issuer}" says "{grant.subject}" can '
            f'"{grant.capability.value}" on {list(grant.task_scope)!r}'
            f" with delegation-depth {grant.delegation_depth}{parent};"
            f" lease={list(grant.lease_scope)!r};"
            f" worktree={list(grant.worktree_scope)!r};"
            f" path={list(grant.path_scope)!r};"
            f" fence={grant.fencing_epoch!r};"
            f" proof-authority={list(grant.proof_authorities)!r};"
            f" override={list(grant.override_scopes)!r};"
            f" valid=[{grant.not_before_ms},{grant.expires_at_ms!r})."
        )
    for revocation in policy.revocations:
        lines.append(
            f'"{revocation.issuer}" revokes "{revocation.target_statement_id}" '
            f"at {revocation.effective_at_ms} because {revocation.reason!r}."
        )
    lines.append(
        f'query "{request.principal}" can "{request.capability.value}" '
        f'on "{request.task_id}" at {request.evaluated_at_ms}; '
        f"lease={request.lease_id!r}; worktree={request.worktree_id!r}; "
        f"path={request.path!r}; fence={request.fencing_epoch!r}; "
        f"proof-authority={request.proof_authority!r}; "
        f"override={request.override_scope!r}."
    )
    return "\n".join(lines) + "\n"


render_secpal = render_secpal_policy


@dataclass(frozen=True)
class AuthorizationFixture(CanonicalContract):
    SCHEMA = AUTHORIZATION_FIXTURE_SCHEMA

    fixture_id: str
    category: str
    policy: AuthorizationPolicy
    request: AuthorizationRequest
    expected_verdict: AuthorizationVerdict

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixture_id", _text(self.fixture_id, "fixture_id"))
        object.__setattr__(self, "category", _text(self.category, "category"))
        if not isinstance(self.policy, AuthorizationPolicy):
            raise AuthorizationValidationError(
                "fixture policy must be an AuthorizationPolicy"
            )
        if not isinstance(self.request, AuthorizationRequest):
            raise AuthorizationValidationError(
                "fixture request must be an AuthorizationRequest"
            )
        object.__setattr__(
            self,
            "expected_verdict",
            _enum(
                self.expected_verdict,
                AuthorizationVerdict,
                "expected_verdict",
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "authorization_logic_version": AUTHORIZATION_LOGIC_VERSION,
            "fixture_id": self.fixture_id,
            "category": self.category,
            "policy": self.policy,
            "request": self.request,
            "expected_verdict": self.expected_verdict,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizationFixture":
        if not isinstance(payload, Mapping):
            raise AuthorizationValidationError(
                "authorization fixture must be an object"
            )
        _schema(payload, cls.SCHEMA)
        raw_policy = payload.get("policy")
        raw_request = payload.get("request")
        if not isinstance(raw_policy, (AuthorizationPolicy, Mapping)):
            raise AuthorizationValidationError(
                "fixture policy must be an authorization policy object"
            )
        if not isinstance(raw_request, (AuthorizationRequest, Mapping)):
            raise AuthorizationValidationError(
                "fixture request must be an authorization request object"
            )
        result = cls(
            fixture_id=payload.get("fixture_id", ""),
            category=payload.get("category", ""),
            policy=(
                raw_policy
                if isinstance(raw_policy, AuthorizationPolicy)
                else AuthorizationPolicy.from_dict(raw_policy)
            ),
            request=(
                raw_request
                if isinstance(raw_request, AuthorizationRequest)
                else AuthorizationRequest.from_dict(raw_request)
            ),
            expected_verdict=payload.get("expected_verdict", ""),
        )
        _identity(payload, result.content_id, "authorization fixture")
        return result


def _fixture_policy(
    *,
    revocations: tuple[Revocation, ...] = (),
    epoch: int = 7,
) -> AuthorizationPolicy:
    root = AuthorizationGrant(
        statement_id="root-execute",
        issuer="root",
        subject="supervisor",
        capability=Capability.EXECUTE_TASK,
        task_scope=("REF-284",),
        delegation_depth=2,
        lease_scope=("lease-current",),
        worktree_scope=("tree-284",),
        path_scope=("ipfs_datasets_py",),
        fencing_epoch=7,
        not_before_ms=100,
        expires_at_ms=2_000,
    )
    worker = AuthorizationGrant(
        statement_id="worker-execute",
        issuer="supervisor",
        subject="worker",
        capability=Capability.EXECUTE_TASK,
        task_scope=("REF-284",),
        delegation_depth=1,
        parent_statement_id=root.statement_id,
        lease_scope=("lease-current",),
        worktree_scope=("tree-284",),
        path_scope=("ipfs_datasets_py/ipfs_accelerate_py",),
        fencing_epoch=7,
        not_before_ms=200,
        expires_at_ms=1_800,
    )
    return AuthorizationPolicy(
        policy_id="supervisor-conformance",
        version="1",
        trusted_roots=("root",),
        grants=(root, worker),
        revocations=revocations,
        current_fencing_epochs={"REF-284": epoch},
        current_lease_ids={"REF-284": "lease-current"},
    )


def default_authorization_fixtures() -> tuple[AuthorizationFixture, ...]:
    """Return the reviewed positive and adversarial policy fixture set."""

    base = dict(
        principal="worker",
        capability=Capability.EXECUTE_TASK,
        task_id="REF-284",
        lease_id="lease-current",
        worktree_id="tree-284",
        path="ipfs_datasets_py/ipfs_accelerate_py/module.py",
        fencing_epoch=7,
    )
    revoked_policy = _fixture_policy(
        revocations=(
            Revocation(
                revocation_id="revoke-worker",
                issuer="supervisor",
                target_statement_id="worker-execute",
                effective_at_ms=800,
                reason="worker replaced",
            ),
        )
    )
    return (
        AuthorizationFixture(
            "positive@1",
            "positive",
            _fixture_policy(),
            AuthorizationRequest(**base, evaluated_at_ms=1_000),
            AuthorizationVerdict.PERMIT,
        ),
        AuthorizationFixture(
            "negative@1",
            "negative",
            _fixture_policy(),
            AuthorizationRequest(
                **{**base, "task_id": "REF-other"}, evaluated_at_ms=1_000
            ),
            AuthorizationVerdict.DENY,
        ),
        AuthorizationFixture(
            "revocation@1",
            "revocation",
            revoked_policy,
            AuthorizationRequest(**base, evaluated_at_ms=1_000),
            AuthorizationVerdict.DENY,
        ),
        AuthorizationFixture(
            "confused-deputy@1",
            "confused_deputy",
            _fixture_policy(),
            AuthorizationRequest(
                **{**base, "principal": "deputy"}, evaluated_at_ms=1_000
            ),
            AuthorizationVerdict.DENY,
        ),
        AuthorizationFixture(
            "stale-lease@1",
            "stale_lease",
            _fixture_policy(epoch=8),
            AuthorizationRequest(**base, evaluated_at_ms=1_000),
            AuthorizationVerdict.DENY,
        ),
    )


DEFAULT_AUTHORIZATION_FIXTURES = default_authorization_fixtures()
AUTHORIZATION_CONFORMANCE_FIXTURES = DEFAULT_AUTHORIZATION_FIXTURES


@dataclass(frozen=True)
class EngineConformanceReceipt(CanonicalContract):
    SCHEMA = AUTHORIZATION_ENGINE_RECEIPT_SCHEMA

    engine: AuthorizationEngine
    status: ConformanceStatus
    fixture_set_identity: str
    executable_path: str
    executable_identity: str
    executable_version: str
    checked_fixture_ids: tuple[str, ...]
    disagreements: tuple[str, ...]
    duration_ms: int
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "engine", _enum(self.engine, AuthorizationEngine, "engine")
        )
        object.__setattr__(
            self, "status", _enum(self.status, ConformanceStatus, "status")
        )
        for name in (
            "fixture_set_identity",
            "executable_path",
            "executable_identity",
            "executable_version",
            "reason",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "checked_fixture_ids",
            _strings(self.checked_fixture_ids, "checked_fixture_ids"),
        )
        object.__setattr__(
            self, "disagreements", _strings(self.disagreements, "disagreements")
        )
        object.__setattr__(
            self, "duration_ms", _nonnegative(self.duration_ms, "duration_ms")
        )
        if self.status is ConformanceStatus.PASSED and (
            self.disagreements
            or set(self.checked_fixture_ids)
            != {item.fixture_id for item in DEFAULT_AUTHORIZATION_FIXTURES}
            or self.fixture_set_identity != _fixture_set_identity()
        ):
            raise AuthorizationValidationError(
                "passing conformance requires every fixture and no disagreement"
            )

    @property
    def passed(self) -> bool:
        return self.status is ConformanceStatus.PASSED

    def _payload(self) -> dict[str, Any]:
        return {
            "authorization_logic_version": AUTHORIZATION_LOGIC_VERSION,
            "engine": self.engine,
            "status": self.status,
            "fixture_set_identity": self.fixture_set_identity,
            "executable_path": self.executable_path,
            "executable_identity": self.executable_identity,
            "executable_version": self.executable_version,
            "checked_fixture_ids": self.checked_fixture_ids,
            "disagreements": self.disagreements,
            "duration_ms": self.duration_ms,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EngineConformanceReceipt":
        if not isinstance(payload, Mapping):
            raise AuthorizationValidationError(
                "engine conformance receipt must be an object"
            )
        _schema(payload, cls.SCHEMA)
        result = cls(
            engine=payload.get("engine", ""),
            status=payload.get("status", ""),
            fixture_set_identity=payload.get("fixture_set_identity", ""),
            executable_path=payload.get("executable_path", ""),
            executable_identity=payload.get("executable_identity", ""),
            executable_version=payload.get("executable_version", ""),
            checked_fixture_ids=tuple(payload.get("checked_fixture_ids") or ()),
            disagreements=tuple(payload.get("disagreements") or ()),
            duration_ms=payload.get("duration_ms", -1),
            reason=payload.get("reason", ""),
        )
        _identity(payload, result.content_id, "engine conformance receipt")
        return result


@dataclass(frozen=True)
class EngineCapability(CanonicalContract):
    SCHEMA = AUTHORIZATION_ENGINE_CAPABILITY_SCHEMA

    engine: AuthorizationEngine
    status: EngineSupportStatus
    reason: str
    executable_path: str | None = None
    executable_version: str | None = None
    conformance_receipt: EngineConformanceReceipt | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "engine", _enum(self.engine, AuthorizationEngine, "engine")
        )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, EngineSupportStatus, "engine support status"),
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if self.status is EngineSupportStatus.CONFORMANT:
            if (
                self.conformance_receipt is None
                or not self.conformance_receipt.passed
                or self.conformance_receipt.engine is not self.engine
                or self.conformance_receipt.executable_path != self.executable_path
                or self.conformance_receipt.executable_version
                != self.executable_version
                or not self.executable_path
                or not self.executable_version
            ):
                raise AuthorizationValidationError(
                    "conformant engine requires its exact passing receipt"
                )
        elif self.conformance_receipt is not None and self.conformance_receipt.passed:
            raise AuthorizationValidationError(
                "a passing conformance receipt cannot be reported as unsupported"
            )

    @property
    def supported(self) -> bool:
        return self.status is EngineSupportStatus.CONFORMANT

    @property
    def available(self) -> bool:
        return self.supported

    def _payload(self) -> dict[str, Any]:
        return {
            "authorization_logic_version": AUTHORIZATION_LOGIC_VERSION,
            "engine": self.engine,
            "status": self.status,
            "supported": self.supported,
            "reason": self.reason,
            "executable_path": self.executable_path,
            "executable_version": self.executable_version,
            "conformance_receipt": self.conformance_receipt,
            "discovery_is_support": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EngineCapability":
        if not isinstance(payload, Mapping):
            raise AuthorizationValidationError("engine capability must be an object")
        _schema(payload, cls.SCHEMA)
        if payload.get("discovery_is_support") not in (None, False):
            raise AuthorizationValidationError(
                "engine discovery cannot be represented as support"
            )
        raw_receipt = payload.get("conformance_receipt")
        result = cls(
            engine=payload.get("engine", ""),
            status=payload.get("status", ""),
            reason=payload.get("reason", ""),
            executable_path=payload.get("executable_path"),
            executable_version=payload.get("executable_version"),
            conformance_receipt=(
                raw_receipt
                if isinstance(raw_receipt, EngineConformanceReceipt)
                else (
                    EngineConformanceReceipt.from_dict(raw_receipt)
                    if isinstance(raw_receipt, Mapping)
                    else None
                )
            ),
        )
        claimed = payload.get("supported")
        if claimed is not None and claimed is not result.supported:
            raise AuthorizationValidationError(
                "supported projection does not match conformance status"
            )
        _identity(payload, result.content_id, "engine capability")
        return result


CommandRunner = Callable[[CommandRequest], CommandResult | Mapping[str, Any]]
ExecutableFinder = Callable[[str], str | None]


def _command_result(value: CommandResult | Mapping[str, Any]) -> CommandResult:
    if isinstance(value, CommandResult):
        return value
    if not isinstance(value, Mapping):
        return CommandResult(returncode=None, error="malformed runner result")
    return CommandResult(
        returncode=value.get("returncode"),
        stdout=str(value.get("stdout") or ""),
        stderr=str(value.get("stderr") or ""),
        timed_out=bool(value.get("timed_out", False)),
        error=str(value["error"]) if value.get("error") else None,
        output_truncated=bool(value.get("output_truncated", False)),
    )


def _executable_identity(path: Path, maximum_bytes: int) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > maximum_bytes:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


class AuthorizationEngineAdapter:
    """Bounded external checker which remains shadow-only after conformance."""

    engine: AuthorizationEngine
    executable_candidates: tuple[str, ...]
    file_suffix: str

    def __init__(
        self,
        *,
        which: ExecutableFinder | None = None,
        command_runner: CommandRunner | None = None,
        timeout_seconds: float = DEFAULT_ENGINE_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_ENGINE_OUTPUT_BYTES,
        max_executable_bytes: int = DEFAULT_MAX_EXECUTABLE_BYTES,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise AuthorizationValidationError("timeout_seconds must be positive")
        self.which = which or shutil.which
        self.command_runner = command_runner or _bounded_command_runner
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = _nonnegative(max_output_bytes, "max_output_bytes")
        self.max_executable_bytes = _nonnegative(
            max_executable_bytes, "max_executable_bytes"
        )
        if self.max_output_bytes == 0 or self.max_executable_bytes == 0:
            raise AuthorizationValidationError("engine byte bounds must be positive")
        self.monotonic = monotonic or time.monotonic
        self._capability: EngineCapability | None = None

    def _discover(self) -> str | None:
        for candidate in self.executable_candidates:
            try:
                path = self.which(candidate)
            except BaseException:
                path = None
            if path:
                return str(Path(path).resolve())
        return None

    def _run(self, request: CommandRequest) -> CommandResult:
        try:
            return _command_result(self.command_runner(request))
        except BaseException as exc:
            return CommandResult(returncode=None, error=type(exc).__name__)

    def render(self, policy: AuthorizationPolicy, request: AuthorizationRequest) -> str:
        return (
            render_datalog_policy(policy, request)
            if self.engine is AuthorizationEngine.DATALOG
            else render_secpal_policy(policy, request)
        )

    def _command(self, executable: str, model_path: str) -> tuple[str, ...]:
        return (executable, model_path)

    @staticmethod
    def _parse(result: CommandResult) -> AuthorizationVerdict | None:
        if result.timed_out or result.error or result.returncode != 0:
            return None
        folded = (result.stdout + "\n" + result.stderr).casefold()
        if re.search(r"\b(permit|allowed|authorized|true)\b", folded):
            return AuthorizationVerdict.PERMIT
        if re.search(r"\b(deny|denied|unauthorized|false)\b", folded):
            return AuthorizationVerdict.DENY
        # Soufflé returns success with an empty output relation for denial.
        return AuthorizationVerdict.DENY

    def _execute(
        self,
        executable: str,
        policy: AuthorizationPolicy,
        request: AuthorizationRequest,
    ) -> AuthorizationVerdict | None:
        with tempfile.TemporaryDirectory(prefix=f"authz-{self.engine.value}-") as raw:
            path = Path(raw) / f"policy.{self.file_suffix}"
            path.write_text(self.render(policy, request), encoding="utf-8")
            result = self._run(
                CommandRequest(
                    command=self._command(executable, str(path)),
                    stdin_text=None,
                    cwd=raw,
                    timeout_seconds=self.timeout_seconds,
                    max_output_bytes=self.max_output_bytes,
                )
            )
        return self._parse(result)

    def probe(self, *, run_conformance: bool = True) -> EngineCapability:
        executable = self._discover()
        if executable is None:
            capability = EngineCapability(
                self.engine,
                EngineSupportStatus.UNSUPPORTED,
                "executable not discovered",
            )
            self._capability = capability
            return capability
        executable_id = _executable_identity(
            Path(executable), self.max_executable_bytes
        )
        if executable_id is None:
            capability = EngineCapability(
                self.engine,
                EngineSupportStatus.UNSUPPORTED,
                "executable identity could not be bounded and pinned",
                executable_path=executable,
            )
            self._capability = capability
            return capability
        version_result = self._run(
            CommandRequest(
                command=(executable, "--version"),
                stdin_text=None,
                cwd=None,
                timeout_seconds=min(2.0, self.timeout_seconds),
                max_output_bytes=min(4096, self.max_output_bytes),
            )
        )
        version_text = (version_result.stdout + "\n" + version_result.stderr).strip()
        if (
            version_result.returncode != 0
            or version_result.timed_out
            or version_result.error
            or not version_text
        ):
            capability = EngineCapability(
                self.engine,
                EngineSupportStatus.UNSUPPORTED,
                "executable did not provide a bounded version",
                executable_path=executable,
            )
            self._capability = capability
            return capability
        version = version_text.splitlines()[0][:256]
        if not run_conformance:
            capability = EngineCapability(
                self.engine,
                EngineSupportStatus.UNSUPPORTED,
                "executable discovered but conformance was not run",
                executable_path=executable,
                executable_version=version,
            )
            self._capability = capability
            return capability

        started = self.monotonic()
        checked: list[str] = []
        disagreements: list[str] = []
        timed_out = False
        errored = False
        for fixture in DEFAULT_AUTHORIZATION_FIXTURES:
            observed = self._execute(executable, fixture.policy, fixture.request)
            checked.append(fixture.fixture_id)
            if observed is None:
                errored = True
                disagreements.append(fixture.fixture_id)
            elif observed is not fixture.expected_verdict:
                disagreements.append(fixture.fixture_id)
        duration_ms = max(0, round((self.monotonic() - started) * 1000))
        if disagreements:
            status = (
                ConformanceStatus.TIMED_OUT
                if timed_out
                else ConformanceStatus.ERROR if errored else ConformanceStatus.FAILED
            )
            reason = "external engine did not agree on every authorization fixture"
        else:
            status = ConformanceStatus.PASSED
            reason = "external engine agreed on every authorization fixture"
        receipt = EngineConformanceReceipt(
            engine=self.engine,
            status=status,
            fixture_set_identity=_fixture_set_identity(),
            executable_path=executable,
            executable_identity=executable_id,
            executable_version=version,
            checked_fixture_ids=tuple(checked),
            disagreements=tuple(disagreements),
            duration_ms=duration_ms,
            reason=reason,
        )
        capability = EngineCapability(
            self.engine,
            (
                EngineSupportStatus.CONFORMANT
                if receipt.passed
                else EngineSupportStatus.UNSUPPORTED
            ),
            reason,
            executable_path=executable,
            executable_version=version,
            conformance_receipt=receipt,
        )
        self._capability = capability
        return capability

    capability = probe
    probe_capability = probe

    def evaluate(
        self, policy: AuthorizationPolicy, request: AuthorizationRequest
    ) -> AuthorizationVerdict | None:
        capability = self._capability or self.probe()
        if not capability.supported or not capability.executable_path:
            return None
        return self._execute(capability.executable_path, policy, request)


class DatalogAuthorizationAdapter(AuthorizationEngineAdapter):
    engine = AuthorizationEngine.DATALOG
    executable_candidates = ("souffle",)
    file_suffix = "dl"


class SecPALAuthorizationAdapter(AuthorizationEngineAdapter):
    engine = AuthorizationEngine.SECPAL
    executable_candidates = ("secpal",)
    file_suffix = "secpal"

    def _command(self, executable: str, model_path: str) -> tuple[str, ...]:
        return (executable, "check", model_path)


DatalogEngineAdapter = DatalogAuthorizationAdapter
SecPALEngineAdapter = SecPALAuthorizationAdapter
DEFAULT_AUTHORIZATION_ADAPTER_TYPES = (
    DatalogAuthorizationAdapter,
    SecPALAuthorizationAdapter,
)


def _fixture_set_identity() -> str:
    digest = hashlib.sha256(
        canonical_json_bytes(
            [item.to_dict() for item in DEFAULT_AUTHORIZATION_FIXTURES]
        )
    ).hexdigest()
    return "sha256:" + digest


def probe_authorization_engines(
    adapters: Sequence[AuthorizationEngineAdapter] | None = None,
    *,
    run_conformance: bool = True,
) -> tuple[EngineCapability, ...]:
    selected = (
        tuple(adapters)
        if adapters is not None
        else tuple(kind() for kind in DEFAULT_AUTHORIZATION_ADAPTER_TYPES)
    )
    engines = [item.engine for item in selected]
    if len(engines) != len(set(engines)):
        raise AuthorizationValidationError(
            "authorization engine adapters must be unique"
        )
    return tuple(item.probe(run_conformance=run_conformance) for item in selected)


@dataclass(frozen=True)
class AuthorizationLaneResult(CanonicalContract):
    SCHEMA = AUTHORIZATION_LANE_RESULT_SCHEMA

    engine: AuthorizationEngine
    status: LaneStatus
    reference_verdict: AuthorizationVerdict
    observed_verdict: AuthorizationVerdict | None
    capability_identity: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "engine", _enum(self.engine, AuthorizationEngine, "engine")
        )
        object.__setattr__(
            self, "status", _enum(self.status, LaneStatus, "lane status")
        )
        object.__setattr__(
            self,
            "reference_verdict",
            _enum(
                self.reference_verdict,
                AuthorizationVerdict,
                "reference verdict",
            ),
        )
        if self.observed_verdict is not None:
            object.__setattr__(
                self,
                "observed_verdict",
                _enum(
                    self.observed_verdict,
                    AuthorizationVerdict,
                    "observed verdict",
                ),
            )
        object.__setattr__(
            self,
            "capability_identity",
            _text(self.capability_identity, "capability_identity"),
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))

    def _payload(self) -> dict[str, Any]:
        return {
            "authorization_logic_version": AUTHORIZATION_LOGIC_VERSION,
            "engine": self.engine,
            "status": self.status,
            "reference_verdict": self.reference_verdict,
            "observed_verdict": self.observed_verdict,
            "capability_identity": self.capability_identity,
            "reason": self.reason,
            "shadow_only": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizationLaneResult":
        if not isinstance(payload, Mapping):
            raise AuthorizationValidationError(
                "authorization lane result must be an object"
            )
        _schema(payload, cls.SCHEMA)
        if payload.get("shadow_only") not in (None, True):
            raise AuthorizationValidationError(
                "external authorization lanes must remain shadow-only"
            )
        result = cls(
            engine=payload.get("engine", ""),
            status=payload.get("status", ""),
            reference_verdict=payload.get("reference_verdict", ""),
            observed_verdict=payload.get("observed_verdict"),
            capability_identity=payload.get("capability_identity", ""),
            reason=payload.get("reason", ""),
        )
        _identity(payload, result.content_id, "authorization lane result")
        return result


@dataclass(frozen=True)
class AuthorizationReport(CanonicalContract):
    """Enforcement decision plus non-authoritative external shadow results."""

    SCHEMA = AUTHORIZATION_REPORT_SCHEMA

    decision: AuthorizationDecision
    shadow_results: tuple[AuthorizationLaneResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.decision, AuthorizationDecision):
            raise AuthorizationValidationError(
                "decision must be an AuthorizationDecision"
            )
        results = tuple(self.shadow_results)
        if any(not isinstance(item, AuthorizationLaneResult) for item in results):
            raise AuthorizationValidationError(
                "shadow_results must contain AuthorizationLaneResult values"
            )
        if len({item.engine for item in results}) != len(results):
            raise AuthorizationValidationError("shadow engine results must be unique")
        object.__setattr__(
            self,
            "shadow_results",
            tuple(sorted(results, key=lambda item: item.engine.value)),
        )

    @property
    def permitted(self) -> bool:
        return self.decision.permitted

    @property
    def establishes_generated_code_correctness(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "authorization_logic_version": AUTHORIZATION_LOGIC_VERSION,
            "decision": self.decision,
            "shadow_results": self.shadow_results,
            "enforcement_lane": "deterministic_reference",
            "external_lanes_are_shadow_only": True,
            "permits_action": self.permitted,
            "establishes_generated_code_correctness": False,
            "generated_code_correctness": GeneratedCodeCorrectness.NOT_ESTABLISHED,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizationReport":
        if not isinstance(payload, Mapping):
            raise AuthorizationValidationError("authorization report must be an object")
        _schema(payload, cls.SCHEMA)
        if payload.get("establishes_generated_code_correctness") not in (None, False):
            raise AuthorizationValidationError(
                "authorization cannot establish generated-code correctness"
            )
        if payload.get("external_lanes_are_shadow_only") not in (None, True):
            raise AuthorizationValidationError(
                "external authorization lanes must remain shadow-only"
            )
        if payload.get("enforcement_lane") not in (
            None,
            "",
            "deterministic_reference",
        ):
            raise AuthorizationValidationError(
                "only the deterministic reference lane may enforce authorization"
            )
        raw_decision = payload.get("decision")
        raw_results = payload.get("shadow_results") or ()
        if not isinstance(raw_decision, (AuthorizationDecision, Mapping)):
            raise AuthorizationValidationError(
                "report decision must be an authorization decision object"
            )
        if isinstance(raw_results, (str, bytes, bytearray, Mapping)) or not isinstance(
            raw_results, Iterable
        ):
            raise AuthorizationValidationError(
                "report shadow_results must be a sequence"
            )
        result = cls(
            decision=(
                raw_decision
                if isinstance(raw_decision, AuthorizationDecision)
                else AuthorizationDecision.from_dict(raw_decision)
            ),
            shadow_results=tuple(
                (
                    item
                    if isinstance(item, AuthorizationLaneResult)
                    else AuthorizationLaneResult.from_dict(item)
                )
                for item in raw_results
            ),
        )
        claimed = payload.get("permits_action")
        if claimed is not None and claimed is not result.permitted:
            raise AuthorizationValidationError(
                "permit projection does not match reference decision"
            )
        _identity(payload, result.content_id, "authorization report")
        return result


class AuthorizationChecker:
    """Run deterministic enforcement even when every external lane is absent."""

    def __init__(
        self,
        *,
        evaluator: ReferenceAuthorizationEvaluator | None = None,
        adapters: Sequence[AuthorizationEngineAdapter] | None = None,
    ) -> None:
        self.evaluator = evaluator or ReferenceAuthorizationEvaluator()
        self.adapters = (
            tuple(adapters)
            if adapters is not None
            else tuple(kind() for kind in DEFAULT_AUTHORIZATION_ADAPTER_TYPES)
        )

    def evaluate(
        self, policy: AuthorizationPolicy, request: AuthorizationRequest
    ) -> AuthorizationReport:
        decision = self.evaluator.evaluate(policy, request)
        results: list[AuthorizationLaneResult] = []
        for adapter in self.adapters:
            capability = adapter._capability or adapter.probe()
            if not capability.supported:
                results.append(
                    AuthorizationLaneResult(
                        adapter.engine,
                        LaneStatus.UNSUPPORTED,
                        decision.verdict,
                        None,
                        capability.content_id,
                        capability.reason,
                    )
                )
                continue
            observed = adapter.evaluate(policy, request)
            if observed is None:
                status = LaneStatus.ERROR
                reason = "conformant external lane did not return a verdict"
            elif observed is decision.verdict:
                status = LaneStatus.AGREED
                reason = "external shadow lane agreed with the reference evaluator"
            else:
                status = LaneStatus.DISAGREED
                reason = "external shadow lane disagreed with the reference evaluator"
            results.append(
                AuthorizationLaneResult(
                    adapter.engine,
                    status,
                    decision.verdict,
                    observed,
                    capability.content_id,
                    reason,
                )
            )
        return AuthorizationReport(decision, tuple(results))

    authorize = evaluate
    check = evaluate


AuthorizationPolicyChecker = AuthorizationChecker


def check_authorization(
    policy: AuthorizationPolicy,
    request: AuthorizationRequest,
    *,
    adapters: Sequence[AuthorizationEngineAdapter] | None = None,
) -> AuthorizationReport:
    return AuthorizationChecker(adapters=adapters).evaluate(policy, request)


__all__ = [
    "AUTHORIZATION_CONFORMANCE_FIXTURES",
    "AUTHORIZATION_LOGIC_VERSION",
    "AUTHORIZATION_POLICY_SCHEMA",
    "AUTHORIZATION_REQUEST_SCHEMA",
    "DEFAULT_AUTHORIZATION_ADAPTER_TYPES",
    "DEFAULT_AUTHORIZATION_FIXTURES",
    "LEASE_BOUND_CAPABILITIES",
    "MAX_DELEGATION_DEPTH",
    "ActionRequest",
    "AuthorityPolicy",
    "AuthorizationCapability",
    "AuthorizationChecker",
    "AuthorizationDecision",
    "AuthorizationEngine",
    "AuthorizationEngineAdapter",
    "AuthorizationEvidence",
    "AuthorizationEvaluator",
    "AuthorizationFixture",
    "AuthorizationGrant",
    "AuthorizationLaneResult",
    "AuthorizationPolicy",
    "AuthorizationPolicyChecker",
    "AuthorizationReport",
    "AuthorizationRequest",
    "AuthorizationValidationError",
    "AuthorizationVerdict",
    "Capability",
    "ConformanceStatus",
    "DatalogAuthorizationAdapter",
    "DatalogEngineAdapter",
    "DelegationGrant",
    "DenialReason",
    "EngineCapability",
    "EngineConformanceReceipt",
    "EngineSupportStatus",
    "GeneratedCodeCorrectness",
    "LaneStatus",
    "PolicyEngine",
    "PolicyEvaluator",
    "PolicyGrant",
    "Principal",
    "ReferenceAuthorizationEvaluator",
    "Revocation",
    "SecPALAuthorizationAdapter",
    "SecPALEngineAdapter",
    "authorize",
    "check_authorization",
    "default_authorization_fixtures",
    "evaluate_authorization",
    "probe_authorization_engines",
    "render_datalog",
    "render_datalog_policy",
    "render_secpal",
    "render_secpal_policy",
]
