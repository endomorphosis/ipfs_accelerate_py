"""Risk-selected formal-verification and rollout policy.

This module is the policy boundary between a repository change and the proof
contracts in :mod:`formal_verification_contracts`.  It deliberately contains
no prover integration.  Its responsibilities are to:

* select assurance and fallback requirements from changed paths, AST scopes,
  risk, and invariant classes;
* make disabled, shadow, canary, and enforcement behaviour deterministic;
* evaluate proof and validation results without treating infrastructure
  failures as proof success; and
* represent rollout transitions and emergency overrides as bounded,
  content-addressed receipts.

The policy is fail closed in a blocking mode.  In particular, ``unsupported``,
``unavailable``, ``timed_out``, ``inconclusive``, and missing outcomes never
satisfy an assurance requirement.  A configured fallback may satisfy a
requirement, but only when the matching rule explicitly allows fallback and
every named validation has an explicit passing result.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Iterable, Mapping, Sequence, Tuple

from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    ContractValidationError,
    ProofPlan,
    ProofReceipt,
    ProofVerdict,
    assurance_satisfies,
    canonical_json,
)

POLICY_VERSION = 1
SCHEMA_VERSION = POLICY_VERSION
CHANGED_SCOPE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/changed-proof-scope@1"
PROOF_POLICY_RULE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/proof-policy-rule@1"
PROOF_REQUIREMENT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/proof-requirement@1"
POLICY_SELECTION_SCHEMA = "ipfs_accelerate_py/agent-supervisor/proof-policy-selection@1"
FORMAL_VERIFICATION_POLICY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/formal-verification-policy@1"
)
PROOF_OUTCOME_SCHEMA = "ipfs_accelerate_py/agent-supervisor/proof-policy-outcome@1"
VALIDATION_OUTCOME_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/fallback-validation-outcome@1"
)
OVERRIDE_RECEIPT_SCHEMA = "ipfs_accelerate_py/agent-supervisor/proof-override-receipt@1"
ROLLOUT_TRANSITION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-rollout-transition-receipt@1"
)
REQUIREMENT_GATE_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-requirement-gate-result@1"
)
POLICY_GATE_DECISION_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-policy-gate-decision@1"
)
MERGE_PROOF_GATE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/merge-proof-gate-receipt@1"
)

MAX_SCOPE_ITEMS = 256
MAX_OVERRIDE_LIFETIME_SECONDS = 7 * 24 * 60 * 60
DEFAULT_MAX_OVERRIDE_SECONDS = 24 * 60 * 60


class PolicyValidationError(ContractValidationError):
    """Raised when a policy or policy receipt is malformed."""


class RiskLevel(str, Enum):
    """Ordered change risk used during policy selection."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3,
        }[self]


class InvariantClass(str, Enum):
    """Common invariant classes.

    Rules also accept non-empty strings so independently developed proof
    providers can introduce reviewed invariant classes without changing this
    module.
    """

    STATE_TRANSITION = "state_transition"
    LEASE_SAFETY = "lease_safety"
    DAG_ACYCLICITY = "dag_acyclicity"
    MERGE_IDEMPOTENCE = "merge_idempotence"
    CACHE_KEY_COMPLETENESS = "cache_key_completeness"
    EVIDENCE_FRESHNESS = "evidence_freshness"
    PROJECTION_EQUIVALENCE = "projection_equivalence"
    AUTHORIZATION = "authorization"
    DATA_INTEGRITY = "data_integrity"
    RESOURCE_ISOLATION = "resource_isolation"


class RolloutMode(str, Enum):
    """Operational mode for selected proof requirements."""

    DISABLED = "disabled"
    SHADOW = "shadow"
    CANARY = "canary"
    ENFORCEMENT = "enforcement"

    @property
    def rank(self) -> int:
        return {
            RolloutMode.DISABLED: 0,
            RolloutMode.SHADOW: 1,
            RolloutMode.CANARY: 2,
            RolloutMode.ENFORCEMENT: 3,
        }[self]

    @property
    def blocks(self) -> bool:
        """Whether this effective mode can block a gate."""

        return self in (RolloutMode.CANARY, RolloutMode.ENFORCEMENT)

    def can_promote_to(self, target: "RolloutMode") -> bool:
        """Return whether ``target`` is the next rollout stage."""

        normalized = _enum(target, RolloutMode, "target")
        return normalized.rank == self.rank + 1

    def can_roll_back_to(self, target: "RolloutMode") -> bool:
        """Return whether ``target`` is a less restrictive stage."""

        normalized = _enum(target, RolloutMode, "target")
        return normalized.rank < self.rank


class ProofResultStatus(str, Enum):
    """Bounded semantic outcomes consumed by the policy gate."""

    PROVED = "proved"
    DISPROVED = "disproved"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    TIMED_OUT = "timed_out"
    INCONCLUSIVE = "inconclusive"
    CANCELLED = "cancelled"
    ERROR = "error"
    MISSING = "missing"


def _enum(value: Any, enum_type: type[Enum], field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, Enum):
        value = value.value
    try:
        return enum_type(str(value).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise PolicyValidationError(f"{field_name} must be one of: {allowed}") from exc


def _text(value: Any, field_name: str, *, required: bool = False) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = value.strip()
    else:
        raise PolicyValidationError(f"{field_name} must be a string")
    if required and not result:
        raise PolicyValidationError(f"{field_name} is required")
    return result


def _strings(
    values: Any,
    field_name: str,
    *,
    required: bool = False,
    preserve_order: bool = False,
) -> Tuple[str, ...]:
    if values is None:
        raw: Iterable[Any] = ()
    elif isinstance(values, (str, Enum)):
        raw = (values.value if isinstance(values, Enum) else values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        raw = values
    else:
        raise PolicyValidationError(f"{field_name} must be a sequence of strings")
    result = []
    for item in raw:
        if isinstance(item, Enum):
            item = item.value
        normalized = _text(item, field_name, required=True)
        if normalized not in result:
            result.append(normalized)
    if len(result) > MAX_SCOPE_ITEMS:
        raise PolicyValidationError(
            f"{field_name} exceeds the {MAX_SCOPE_ITEMS}-item policy bound"
        )
    if required and not result:
        raise PolicyValidationError(f"{field_name} must not be empty")
    return tuple(result if preserve_order else sorted(result))


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PolicyValidationError(f"{field_name} must be a mapping")
    # Round-trip through the shared canonical encoder now, rather than waiting
    # until identity calculation, so mutable aliases, floats, and unsupported
    # values cannot enter an otherwise valid frozen policy contract.
    normalized = json.loads(canonical_json(value))
    if not isinstance(normalized, dict):
        raise PolicyValidationError(f"{field_name} must be a mapping")
    return normalized


def _integer(
    value: Any,
    field_name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PolicyValidationError(f"{field_name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = (
            f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        )
        raise PolicyValidationError(f"{field_name} must be {bounds}")
    return value


def _normalize_git_path(value: Any, field_name: str = "path") -> str:
    path = _text(value, field_name, required=True)
    if "\\" in path:
        raise PolicyValidationError(f"{field_name} must use '/' separators")
    if path.startswith("/") or PurePosixPath(path).is_absolute():
        raise PolicyValidationError(f"{field_name} must be repository-relative")
    while path.startswith("./"):
        path = path[2:]
    parts = PurePosixPath(path).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise PolicyValidationError(f"{field_name} must be a safe repository path")
    return "/".join(parts)


def _normalize_path_pattern(value: Any) -> str:
    pattern = _normalize_git_path(value, "path_patterns")
    return pattern


def _path_matches(path: str, pattern: str) -> bool:
    """Match a repository path without allowing ``*`` to cross ``/``.

    Python's :mod:`fnmatch` treats path separators as ordinary characters,
    which would make ``src/*.py`` unexpectedly protect nested files.  This
    translator uses git-style ``**`` semantics: ``**/`` matches zero or more
    directories and a trailing ``/**`` includes the directory itself.
    """

    expression = ["^"]
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if pattern.startswith("**/", index):
            expression.append("(?:.*/)?")
            index += 3
            continue
        if pattern.startswith("/**", index) and index + 3 == len(pattern):
            expression.append("(?:/.*)?")
            index += 3
            continue
        if pattern.startswith("**", index):
            expression.append(".*")
            index += 2
            continue
        if character == "*":
            expression.append("[^/]*")
        elif character == "?":
            expression.append("[^/]")
        elif character == "[":
            closing = pattern.find("]", index + 1)
            if closing < 0:
                expression.append(r"\[")
            else:
                content = pattern[index + 1 : closing]
                if content.startswith("!"):
                    content = "^" + content[1:]
                elif content.startswith("^"):
                    content = "\\" + content
                expression.append("[" + content.replace("\\", r"\\") + "]")
                index = closing
        else:
            expression.append(re.escape(character))
        index += 1
    expression.append("$")
    return re.match("".join(expression), path) is not None


def _timestamp(value: Any, field_name: str) -> tuple[str, datetime]:
    text = _text(value, field_name, required=True)
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise PolicyValidationError(
            f"{field_name} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PolicyValidationError(f"{field_name} must include a timezone")
    parsed = parsed.astimezone(timezone.utc)
    normalized = parsed.isoformat(timespec="microseconds").replace("+00:00", "Z")
    return normalized, parsed


def _now(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise PolicyValidationError("now must include a timezone")
        return value.astimezone(timezone.utc)
    return _timestamp(value, "now")[1]


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise PolicyValidationError(
            f"unsupported schema {supplied!r}; expected {expected}"
        )


@dataclass(frozen=True)
class ChangedScope(CanonicalContract):
    """One changed repository path with its semantic classification."""

    SCHEMA: ClassVar[str] = CHANGED_SCOPE_SCHEMA

    path: str
    ast_scope_ids: Tuple[str, ...] = ()
    risk: RiskLevel = RiskLevel.LOW
    invariant_classes: Tuple[str, ...] = ()
    change_kind: str = "modified"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _normalize_git_path(self.path))
        object.__setattr__(
            self, "ast_scope_ids", _strings(self.ast_scope_ids, "ast_scope_ids")
        )
        object.__setattr__(self, "risk", _enum(self.risk, RiskLevel, "risk"))
        object.__setattr__(
            self,
            "invariant_classes",
            _strings(self.invariant_classes, "invariant_classes"),
        )
        object.__setattr__(
            self,
            "change_kind",
            _text(self.change_kind, "change_kind", required=True).lower(),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def scope_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "policy_contract_version": POLICY_VERSION,
            "path": self.path,
            "ast_scope_ids": self.ast_scope_ids,
            "risk": self.risk,
            "invariant_classes": self.invariant_classes,
            "change_kind": self.change_kind,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ChangedScope":
        _schema(payload, cls.SCHEMA)
        result = cls(
            path=payload.get("path", ""),
            ast_scope_ids=tuple(payload.get("ast_scope_ids") or ()),
            risk=payload.get("risk", RiskLevel.LOW),
            invariant_classes=tuple(payload.get("invariant_classes") or ()),
            change_kind=payload.get("change_kind", "modified"),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("scope_id") or payload.get("content_id")
        if claimed and claimed != result.scope_id:
            raise PolicyValidationError("changed scope content identity does not match")
        return result


ChangeScope = ChangedCodeScope = ChangedScope


@dataclass(frozen=True)
class ProofPolicyRule(CanonicalContract):
    """A conjunctive selector and its required proof/fallback assurance."""

    SCHEMA: ClassVar[str] = PROOF_POLICY_RULE_SCHEMA

    rule_id: str
    required_assurance: AssuranceLevel
    path_patterns: Tuple[str, ...] = ()
    ast_scope_patterns: Tuple[str, ...] = ()
    minimum_risk: RiskLevel = RiskLevel.LOW
    invariant_classes: Tuple[str, ...] = ()
    fallback_validations: Tuple[str, ...] = ()
    allow_fallback: bool = False
    match_all: bool = False
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rule_id", _text(self.rule_id, "rule_id", required=True)
        )
        object.__setattr__(
            self,
            "required_assurance",
            _enum(self.required_assurance, AssuranceLevel, "required_assurance"),
        )
        object.__setattr__(
            self,
            "path_patterns",
            tuple(
                sorted(
                    {
                        _normalize_path_pattern(item)
                        for item in _strings(self.path_patterns, "path_patterns")
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "ast_scope_patterns",
            _strings(self.ast_scope_patterns, "ast_scope_patterns"),
        )
        object.__setattr__(
            self, "minimum_risk", _enum(self.minimum_risk, RiskLevel, "minimum_risk")
        )
        object.__setattr__(
            self,
            "invariant_classes",
            _strings(self.invariant_classes, "invariant_classes"),
        )
        object.__setattr__(
            self,
            "fallback_validations",
            _strings(self.fallback_validations, "fallback_validations"),
        )
        if not isinstance(self.allow_fallback, bool):
            raise PolicyValidationError("allow_fallback must be a boolean")
        if not isinstance(self.match_all, bool):
            raise PolicyValidationError("match_all must be a boolean")
        object.__setattr__(self, "description", _text(self.description, "description"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        has_selector = bool(
            self.path_patterns
            or self.ast_scope_patterns
            or self.invariant_classes
            or self.minimum_risk is not RiskLevel.LOW
        )
        if not has_selector and not self.match_all:
            raise PolicyValidationError(
                "a rule needs a selector or explicit match_all=True"
            )
        if self.allow_fallback and not self.fallback_validations:
            raise PolicyValidationError(
                "allow_fallback requires at least one fallback validation"
            )

    def matches(self, change: ChangedScope) -> bool:
        """Return whether all configured selector dimensions match ``change``."""

        if not isinstance(change, ChangedScope):
            change = ChangedScope.from_dict(change)  # type: ignore[arg-type]
        if change.risk.rank < self.minimum_risk.rank:
            return False
        if self.path_patterns and not any(
            _path_matches(change.path, pattern) for pattern in self.path_patterns
        ):
            return False
        if self.ast_scope_patterns and not any(
            fnmatch.fnmatchcase(scope, pattern)
            for scope in change.ast_scope_ids
            for pattern in self.ast_scope_patterns
        ):
            return False
        if self.invariant_classes and not (
            set(self.invariant_classes) & set(change.invariant_classes)
        ):
            return False
        return True

    def _payload(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "path_patterns": self.path_patterns,
            "ast_scope_patterns": self.ast_scope_patterns,
            "minimum_risk": self.minimum_risk,
            "invariant_classes": self.invariant_classes,
            "required_assurance": self.required_assurance,
            "fallback_validations": self.fallback_validations,
            "allow_fallback": self.allow_fallback,
            "match_all": self.match_all,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofPolicyRule":
        _schema(payload, cls.SCHEMA)
        return cls(
            rule_id=payload.get("rule_id", ""),
            path_patterns=tuple(payload.get("path_patterns") or ()),
            ast_scope_patterns=tuple(payload.get("ast_scope_patterns") or ()),
            minimum_risk=payload.get("minimum_risk", RiskLevel.LOW),
            invariant_classes=tuple(payload.get("invariant_classes") or ()),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
            fallback_validations=tuple(payload.get("fallback_validations") or ()),
            allow_fallback=payload.get("allow_fallback", False),
            match_all=payload.get("match_all", False),
            description=payload.get("description", ""),
            metadata=payload.get("metadata") or {},
        )


FormalVerificationRule = PolicyRule = ProofPolicyRule


@dataclass(frozen=True)
class ProofRequirement(CanonicalContract):
    """The conservative merge of every rule matching one changed scope."""

    SCHEMA: ClassVar[str] = PROOF_REQUIREMENT_SCHEMA

    scope_id: str
    path: str
    ast_scope_ids: Tuple[str, ...]
    risk: RiskLevel
    invariant_classes: Tuple[str, ...]
    matched_rule_ids: Tuple[str, ...]
    required_assurance: AssuranceLevel
    fallback_validations: Tuple[str, ...] = ()
    allow_fallback: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "scope_id", _text(self.scope_id, "scope_id", required=True)
        )
        object.__setattr__(self, "path", _normalize_git_path(self.path))
        object.__setattr__(
            self, "ast_scope_ids", _strings(self.ast_scope_ids, "ast_scope_ids")
        )
        object.__setattr__(self, "risk", _enum(self.risk, RiskLevel, "risk"))
        object.__setattr__(
            self,
            "invariant_classes",
            _strings(self.invariant_classes, "invariant_classes"),
        )
        object.__setattr__(
            self,
            "matched_rule_ids",
            _strings(self.matched_rule_ids, "matched_rule_ids", required=True),
        )
        object.__setattr__(
            self,
            "required_assurance",
            _enum(self.required_assurance, AssuranceLevel, "required_assurance"),
        )
        object.__setattr__(
            self,
            "fallback_validations",
            _strings(self.fallback_validations, "fallback_validations"),
        )
        if not isinstance(self.allow_fallback, bool):
            raise PolicyValidationError("allow_fallback must be a boolean")
        if self.allow_fallback and not self.fallback_validations:
            raise PolicyValidationError(
                "allow_fallback requires at least one fallback validation"
            )

    @property
    def requirement_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "path": self.path,
            "ast_scope_ids": self.ast_scope_ids,
            "risk": self.risk,
            "invariant_classes": self.invariant_classes,
            "matched_rule_ids": self.matched_rule_ids,
            "required_assurance": self.required_assurance,
            "fallback_validations": self.fallback_validations,
            "allow_fallback": self.allow_fallback,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofRequirement":
        _schema(payload, cls.SCHEMA)
        result = cls(
            scope_id=payload.get("scope_id", ""),
            path=payload.get("path", ""),
            ast_scope_ids=tuple(payload.get("ast_scope_ids") or ()),
            risk=payload.get("risk", RiskLevel.LOW),
            invariant_classes=tuple(payload.get("invariant_classes") or ()),
            matched_rule_ids=tuple(payload.get("matched_rule_ids") or ()),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
            fallback_validations=tuple(payload.get("fallback_validations") or ()),
            allow_fallback=payload.get("allow_fallback", False),
        )
        claimed = payload.get("requirement_id") or payload.get("content_id")
        if claimed and claimed != result.requirement_id:
            raise PolicyValidationError("requirement content identity does not match")
        return result


@dataclass(frozen=True)
class PolicySelection(CanonicalContract):
    """Deterministic policy selection for an exact repository tree."""

    SCHEMA: ClassVar[str] = POLICY_SELECTION_SCHEMA

    policy_id: str
    repository_tree_id: str
    rollout_mode: RolloutMode
    requirements: Tuple[ProofRequirement, ...] = ()
    unprotected_scope_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", required=True)
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, "repository_tree_id"),
        )
        object.__setattr__(
            self, "rollout_mode", _enum(self.rollout_mode, RolloutMode, "rollout_mode")
        )
        requirements = tuple(
            sorted(
                (
                    (
                        item
                        if isinstance(item, ProofRequirement)
                        else ProofRequirement.from_dict(item)
                    )
                    for item in self.requirements
                ),
                key=lambda item: item.requirement_id,
            )
        )
        if len({item.scope_id for item in requirements}) != len(requirements):
            raise PolicyValidationError("each changed scope may have one requirement")
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(
            self,
            "unprotected_scope_ids",
            _strings(self.unprotected_scope_ids, "unprotected_scope_ids"),
        )

    @property
    def selection_id(self) -> str:
        return self.content_id

    @property
    def required_assurance(self) -> AssuranceLevel:
        if not self.requirements:
            return AssuranceLevel.UNVERIFIED
        return max(
            (item.required_assurance for item in self.requirements),
            key=lambda item: item.rank,
        )

    @property
    def fallback_validations(self) -> Tuple[str, ...]:
        return tuple(
            sorted(
                {
                    validation
                    for requirement in self.requirements
                    for validation in requirement.fallback_validations
                }
            )
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "repository_tree_id": self.repository_tree_id,
            "rollout_mode": self.rollout_mode,
            "requirements": self.requirements,
            "unprotected_scope_ids": self.unprotected_scope_ids,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicySelection":
        _schema(payload, cls.SCHEMA)
        result = cls(
            policy_id=payload.get("policy_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            rollout_mode=payload.get("rollout_mode", RolloutMode.DISABLED),
            requirements=tuple(
                (
                    item
                    if isinstance(item, ProofRequirement)
                    else ProofRequirement.from_dict(item)
                )
                for item in payload.get("requirements") or ()
            ),
            unprotected_scope_ids=tuple(payload.get("unprotected_scope_ids") or ()),
        )
        claimed = payload.get("selection_id") or payload.get("content_id")
        if claimed and claimed != result.selection_id:
            raise PolicyValidationError("selection content identity does not match")
        return result


def _rule(value: Any) -> ProofPolicyRule:
    if isinstance(value, ProofPolicyRule):
        return value
    if isinstance(value, Mapping):
        return ProofPolicyRule.from_dict(value)
    raise PolicyValidationError("rules must contain ProofPolicyRule values")


@dataclass(frozen=True)
class FormalVerificationPolicy(CanonicalContract):
    """Versioned selection, fallback, canary, and promotion configuration."""

    SCHEMA: ClassVar[str] = FORMAL_VERIFICATION_POLICY_SCHEMA

    name: str
    version: str
    rollout_mode: RolloutMode
    rules: Tuple[ProofPolicyRule, ...]
    canary_path_patterns: Tuple[str, ...] = ()
    canary_percent: int = 0
    canary_salt: str = ""
    max_override_seconds: int = DEFAULT_MAX_OVERRIDE_SECONDS
    minimum_promotion_observations: int = 1
    maximum_promotion_blocking_results: int = 0
    maximum_promotion_overrides: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "name", required=True))
        object.__setattr__(
            self, "version", _text(self.version, "version", required=True)
        )
        object.__setattr__(
            self, "rollout_mode", _enum(self.rollout_mode, RolloutMode, "rollout_mode")
        )
        normalized_rules = tuple(
            sorted((_rule(item) for item in self.rules), key=lambda x: x.rule_id)
        )
        if len({item.rule_id for item in normalized_rules}) != len(normalized_rules):
            raise PolicyValidationError("policy rule_id values must be unique")
        object.__setattr__(self, "rules", normalized_rules)
        object.__setattr__(
            self,
            "canary_path_patterns",
            tuple(
                sorted(
                    {
                        _normalize_path_pattern(item)
                        for item in _strings(
                            self.canary_path_patterns, "canary_path_patterns"
                        )
                    }
                )
            ),
        )
        object.__setattr__(
            self,
            "canary_percent",
            _integer(self.canary_percent, "canary_percent", maximum=100),
        )
        object.__setattr__(self, "canary_salt", _text(self.canary_salt, "canary_salt"))
        object.__setattr__(
            self,
            "max_override_seconds",
            _integer(
                self.max_override_seconds,
                "max_override_seconds",
                minimum=1,
                maximum=MAX_OVERRIDE_LIFETIME_SECONDS,
            ),
        )
        for name in (
            "minimum_promotion_observations",
            "maximum_promotion_blocking_results",
            "maximum_promotion_overrides",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=0)
            )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def policy_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "policy_contract_version": POLICY_VERSION,
            "name": self.name,
            "version": self.version,
            "rollout_mode": self.rollout_mode,
            "rules": self.rules,
            "canary_path_patterns": self.canary_path_patterns,
            "canary_percent": self.canary_percent,
            "canary_salt": self.canary_salt,
            "max_override_seconds": self.max_override_seconds,
            "minimum_promotion_observations": self.minimum_promotion_observations,
            "maximum_promotion_blocking_results": (
                self.maximum_promotion_blocking_results
            ),
            "maximum_promotion_overrides": self.maximum_promotion_overrides,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FormalVerificationPolicy":
        _schema(payload, cls.SCHEMA)
        result = cls(
            name=payload.get("name", ""),
            version=payload.get("version", ""),
            rollout_mode=payload.get("rollout_mode", RolloutMode.DISABLED),
            rules=tuple(_rule(item) for item in payload.get("rules") or ()),
            canary_path_patterns=tuple(payload.get("canary_path_patterns") or ()),
            canary_percent=payload.get("canary_percent", 0),
            canary_salt=payload.get("canary_salt", ""),
            max_override_seconds=payload.get(
                "max_override_seconds", DEFAULT_MAX_OVERRIDE_SECONDS
            ),
            minimum_promotion_observations=payload.get(
                "minimum_promotion_observations", 1
            ),
            maximum_promotion_blocking_results=payload.get(
                "maximum_promotion_blocking_results", 0
            ),
            maximum_promotion_overrides=payload.get("maximum_promotion_overrides", 0),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("policy_id") or payload.get("content_id")
        if claimed and claimed != result.policy_id:
            raise PolicyValidationError("policy content identity does not match")
        return result

    def select(
        self,
        changes: Iterable[ChangedScope | Mapping[str, Any]],
        *,
        repository_tree_id: str = "",
    ) -> PolicySelection:
        """Select and conservatively merge requirements for ``changes``."""

        normalized_changes = tuple(
            sorted(
                (
                    (
                        item
                        if isinstance(item, ChangedScope)
                        else ChangedScope.from_dict(item)
                    )
                    for item in changes
                ),
                key=lambda item: item.scope_id,
            )
        )
        if len({item.scope_id for item in normalized_changes}) != len(
            normalized_changes
        ):
            raise PolicyValidationError("changed scopes must be unique")

        requirements = []
        unprotected = []
        for change in normalized_changes:
            matching = tuple(rule for rule in self.rules if rule.matches(change))
            if not matching:
                unprotected.append(change.scope_id)
                continue
            required = max(
                (rule.required_assurance for rule in matching),
                key=lambda item: item.rank,
            )
            fallback_validations = tuple(
                sorted(
                    {
                        validation
                        for rule in matching
                        for validation in rule.fallback_validations
                    }
                )
            )
            # Every applicable policy constraint must permit fallback.  A
            # permissive broad rule cannot weaken a stricter matching rule.
            allow_fallback = bool(matching) and all(
                rule.allow_fallback for rule in matching
            )
            requirements.append(
                ProofRequirement(
                    scope_id=change.scope_id,
                    path=change.path,
                    ast_scope_ids=change.ast_scope_ids,
                    risk=change.risk,
                    invariant_classes=change.invariant_classes,
                    matched_rule_ids=tuple(rule.rule_id for rule in matching),
                    required_assurance=required,
                    fallback_validations=fallback_validations,
                    allow_fallback=allow_fallback,
                )
            )
        return PolicySelection(
            policy_id=self.policy_id,
            repository_tree_id=repository_tree_id,
            rollout_mode=self.rollout_mode,
            requirements=tuple(requirements),
            unprotected_scope_ids=tuple(unprotected),
        )

    def select_change(
        self,
        *,
        path: str,
        ast_scope_ids: Sequence[str] = (),
        risk: RiskLevel = RiskLevel.LOW,
        invariant_classes: Sequence[str] = (),
        repository_tree_id: str = "",
        change_kind: str = "modified",
    ) -> PolicySelection:
        """Convenience wrapper for selecting a single changed path."""

        return self.select(
            (
                ChangedScope(
                    path=path,
                    ast_scope_ids=tuple(ast_scope_ids),
                    risk=risk,
                    invariant_classes=tuple(invariant_classes),
                    change_kind=change_kind,
                ),
            ),
            repository_tree_id=repository_tree_id,
        )

    def is_canary_selected(self, requirement: ProofRequirement) -> bool:
        """Deterministically select a requirement for blocking canary."""

        if any(
            _path_matches(requirement.path, pattern)
            for pattern in self.canary_path_patterns
        ):
            return True
        if self.canary_percent <= 0:
            return False
        if self.canary_percent >= 100:
            return True
        # Hash the explicit policy, salt, and requirement identities so the
        # bucket remains stable across processes and changes when policy does.
        key = (
            f"{self.policy_id}\0{self.canary_salt}\0" f"{requirement.requirement_id}"
        ).encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % 100
        return bucket < self.canary_percent

    def effective_mode(self, requirement: ProofRequirement) -> RolloutMode:
        """Return the mode applying to one selected requirement."""

        if self.rollout_mode is RolloutMode.CANARY:
            return (
                RolloutMode.CANARY
                if self.is_canary_selected(requirement)
                else RolloutMode.SHADOW
            )
        return self.rollout_mode

    def transition(
        self, receipt: "RolloutTransitionReceipt"
    ) -> "FormalVerificationPolicy":
        """Return a new policy after validating an explicit transition receipt."""

        if not isinstance(receipt, RolloutTransitionReceipt):
            raise PolicyValidationError("rollout transition requires a durable receipt")
        if receipt.policy_id != self.policy_id:
            raise PolicyValidationError("transition receipt policy_id does not match")
        if receipt.from_mode is not self.rollout_mode:
            raise PolicyValidationError("transition receipt source mode does not match")
        target = receipt.target_mode
        if self.rollout_mode.can_roll_back_to(target):
            return replace(self, rollout_mode=target)
        if not self.rollout_mode.can_promote_to(target):
            raise PolicyValidationError(
                "rollout promotion must advance exactly one mode"
            )
        if self.rollout_mode is not RolloutMode.DISABLED:
            if receipt.observation_count < self.minimum_promotion_observations:
                raise PolicyValidationError("insufficient observations for promotion")
            if receipt.blocking_result_count > self.maximum_promotion_blocking_results:
                raise PolicyValidationError("blocking results exceed promotion policy")
            if receipt.override_count > self.maximum_promotion_overrides:
                raise PolicyValidationError("overrides exceed promotion policy")
            if not receipt.evidence_receipt_ids:
                raise PolicyValidationError(
                    "promotion requires durable evidence receipts"
                )
        return replace(self, rollout_mode=target)

    def evaluate_gate(
        self,
        selection: PolicySelection,
        outcomes: Any = None,
        *,
        validations: Any = None,
        override: "OverrideReceipt | None" = None,
        now: datetime | str | None = None,
    ) -> "PolicyGateDecision":
        """Evaluate an exact selection using fail-closed proof semantics."""

        if selection.policy_id != self.policy_id:
            raise PolicyValidationError("selection was produced by a different policy")
        normalized_outcomes = _outcome_map(outcomes, selection.requirements)
        normalized_validations = _validation_map(validations)
        current_time = _now(now)
        override_valid = False
        override_reasons: Tuple[str, ...] = ()
        if override is not None:
            override_reasons = override.invalid_reasons(
                policy=self,
                selection=selection,
                now=current_time,
            )
            override_valid = not override_reasons

        results = []
        for requirement in selection.requirements:
            mode = self.effective_mode(requirement)
            outcome = normalized_outcomes.get(
                requirement.requirement_id,
                ProofOutcome.missing(requirement.requirement_id),
            )
            proof_satisfied = (
                outcome.status is ProofResultStatus.PROVED
                and bool(outcome.receipt_id)
                and assurance_satisfies(
                    outcome.authoritative_assurance,
                    requirement.required_assurance,
                )
            )
            missing_validations = tuple(
                validation_id
                for validation_id in requirement.fallback_validations
                if validation_id not in normalized_validations
                or not normalized_validations[validation_id].passed
            )
            fallback_satisfied = (
                not proof_satisfied
                and requirement.allow_fallback
                and bool(requirement.fallback_validations)
                and not missing_validations
            )
            requirement_satisfied = proof_satisfied or fallback_satisfied
            reasons = []
            if mode is RolloutMode.DISABLED:
                reasons.append("proof_policy_disabled")
                allowed = True
                would_block = False
            else:
                if proof_satisfied:
                    reasons.append("required_assurance_satisfied")
                elif fallback_satisfied:
                    reasons.append("explicit_fallback_validations_satisfied")
                else:
                    reasons.append(f"proof_{outcome.status.value}")
                    if outcome.status is ProofResultStatus.PROVED:
                        if not outcome.receipt_id:
                            reasons.append("proof_receipt_missing")
                        else:
                            reasons.append("required_assurance_not_satisfied")
                    if not requirement.allow_fallback:
                        reasons.append("fallback_not_permitted")
                    elif missing_validations:
                        reasons.append("fallback_validation_missing_or_failed")

                would_block = not requirement_satisfied
                if mode is RolloutMode.SHADOW:
                    allowed = True
                    if would_block:
                        reasons.append("shadow_would_block")
                else:
                    allowed = requirement_satisfied
                    if not allowed and override_valid and override is not None:
                        if override.covers(requirement, mode):
                            allowed = True
                            reasons.append("bounded_override_applied")
                        else:
                            reasons.append("override_does_not_cover_requirement")
            results.append(
                RequirementGateResult(
                    requirement_id=requirement.requirement_id,
                    path=requirement.path,
                    effective_mode=mode,
                    proof_status=outcome.status,
                    required_assurance=requirement.required_assurance,
                    authoritative_assurance=outcome.authoritative_assurance,
                    proof_receipt_id=outcome.receipt_id,
                    proof_satisfied=proof_satisfied,
                    fallback_satisfied=fallback_satisfied,
                    missing_or_failed_validations=missing_validations,
                    requirement_satisfied=requirement_satisfied,
                    allowed=allowed,
                    would_block=would_block,
                    override_receipt_id=(
                        override.receipt_id
                        if override is not None
                        and override_valid
                        and "bounded_override_applied" in reasons
                        else ""
                    ),
                    reason_codes=tuple(reasons),
                )
            )

        return PolicyGateDecision(
            policy_id=self.policy_id,
            selection_id=selection.selection_id,
            repository_tree_id=selection.repository_tree_id,
            rollout_mode=self.rollout_mode,
            allowed=all(result.allowed for result in results),
            results=tuple(results),
            override_receipt_id=(
                override.receipt_id
                if override is not None
                and any(result.override_receipt_id for result in results)
                else ""
            ),
            override_rejection_reasons=override_reasons,
        )


RiskSelectedProofPolicy = VerificationPolicy = FormalVerificationPolicy


@dataclass(frozen=True)
class ProofOutcome(CanonicalContract):
    """A proof result projected into policy-safe fields."""

    SCHEMA: ClassVar[str] = PROOF_OUTCOME_SCHEMA

    requirement_id: str
    status: ProofResultStatus
    authoritative_assurance: AssuranceLevel = AssuranceLevel.UNVERIFIED
    receipt_id: str = ""
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requirement_id",
            _text(self.requirement_id, "requirement_id", required=True),
        )
        object.__setattr__(
            self, "status", _enum(self.status, ProofResultStatus, "status")
        )
        object.__setattr__(
            self,
            "authoritative_assurance",
            _enum(
                self.authoritative_assurance,
                AssuranceLevel,
                "authoritative_assurance",
            ),
        )
        object.__setattr__(self, "receipt_id", _text(self.receipt_id, "receipt_id"))
        object.__setattr__(self, "reason_code", _text(self.reason_code, "reason_code"))

    @property
    def assurance(self) -> AssuranceLevel:
        return self.authoritative_assurance

    def _payload(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "status": self.status,
            "authoritative_assurance": self.authoritative_assurance,
            "receipt_id": self.receipt_id,
            "reason_code": self.reason_code,
        }

    @classmethod
    def missing(cls, requirement_id: str) -> "ProofOutcome":
        return cls(requirement_id=requirement_id, status=ProofResultStatus.MISSING)

    @classmethod
    def from_receipt(cls, requirement_id: str, receipt: ProofReceipt) -> "ProofOutcome":
        if not isinstance(receipt, ProofReceipt):
            raise PolicyValidationError("receipt must be a ProofReceipt")
        status = {
            ProofVerdict.PROVED: ProofResultStatus.PROVED,
            ProofVerdict.DISPROVED: ProofResultStatus.DISPROVED,
            ProofVerdict.INCONCLUSIVE: ProofResultStatus.INCONCLUSIVE,
            ProofVerdict.UNSUPPORTED: ProofResultStatus.UNSUPPORTED,
            ProofVerdict.ERROR: ProofResultStatus.ERROR,
            ProofVerdict.CANCELLED: ProofResultStatus.CANCELLED,
        }[receipt.verdict]
        return cls(
            requirement_id=requirement_id,
            status=status,
            authoritative_assurance=receipt.authoritative_assurance,
            receipt_id=receipt.receipt_id,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofOutcome":
        _schema(payload, cls.SCHEMA)
        return cls(
            requirement_id=payload.get("requirement_id", ""),
            status=payload.get("status", ProofResultStatus.MISSING),
            authoritative_assurance=payload.get(
                "authoritative_assurance", AssuranceLevel.UNVERIFIED
            ),
            receipt_id=payload.get("receipt_id", ""),
            reason_code=payload.get("reason_code", ""),
        )


ProofResult = ProofGateOutcome = ProofOutcome


@dataclass(frozen=True)
class ValidationOutcome(CanonicalContract):
    """Explicit outcome for one configured fallback validation."""

    SCHEMA: ClassVar[str] = VALIDATION_OUTCOME_SCHEMA

    validation_id: str
    passed: bool
    receipt_id: str = ""
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "validation_id",
            _text(self.validation_id, "validation_id", required=True),
        )
        if not isinstance(self.passed, bool):
            raise PolicyValidationError("passed must be a boolean")
        object.__setattr__(self, "receipt_id", _text(self.receipt_id, "receipt_id"))
        object.__setattr__(self, "reason_code", _text(self.reason_code, "reason_code"))

    def _payload(self) -> dict[str, Any]:
        return {
            "validation_id": self.validation_id,
            "passed": self.passed,
            "receipt_id": self.receipt_id,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationOutcome":
        _schema(payload, cls.SCHEMA)
        return cls(
            validation_id=payload.get("validation_id", ""),
            passed=payload.get("passed", False),
            receipt_id=payload.get("receipt_id", ""),
            reason_code=payload.get("reason_code", ""),
        )


FallbackValidationResult = ValidationResult = ValidationOutcome


def _outcome_map(
    values: Any, requirements: Sequence[ProofRequirement]
) -> dict[str, ProofOutcome]:
    if values is None:
        return {}
    if isinstance(values, ProofReceipt):
        if len(requirements) != 1:
            raise PolicyValidationError(
                "a single proof receipt can only evaluate one requirement"
            )
        outcome = ProofOutcome.from_receipt(requirements[0].requirement_id, values)
        return {outcome.requirement_id: outcome}
    if isinstance(values, ProofOutcome):
        return {values.requirement_id: values}
    result: dict[str, ProofOutcome] = {}
    if isinstance(values, Mapping):
        raw_items = values.items()
        for key, raw in raw_items:
            requirement_id = str(key)
            if isinstance(raw, ProofOutcome):
                outcome = raw
            elif isinstance(raw, ProofReceipt):
                outcome = ProofOutcome.from_receipt(requirement_id, raw)
            elif isinstance(raw, Mapping):
                outcome = ProofOutcome(
                    requirement_id=raw.get("requirement_id", requirement_id),
                    status=raw.get("status", ProofResultStatus.MISSING),
                    authoritative_assurance=raw.get(
                        "authoritative_assurance",
                        raw.get("assurance", AssuranceLevel.UNVERIFIED),
                    ),
                    receipt_id=raw.get("receipt_id", ""),
                    reason_code=raw.get("reason_code", ""),
                )
            else:
                raise PolicyValidationError("proof outcome mapping values are invalid")
            result[outcome.requirement_id] = outcome
        return result
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        for raw in values:
            if not isinstance(raw, ProofOutcome):
                raise PolicyValidationError(
                    "proof outcomes must contain ProofOutcome values"
                )
            result[raw.requirement_id] = raw
        return result
    raise PolicyValidationError("proof outcomes have an unsupported shape")


def _validation_map(values: Any) -> dict[str, ValidationOutcome]:
    if values is None:
        return {}
    if isinstance(values, ValidationOutcome):
        return {values.validation_id: values}
    result: dict[str, ValidationOutcome] = {}
    if isinstance(values, Mapping):
        for key, raw in values.items():
            validation_id = str(key)
            if isinstance(raw, ValidationOutcome):
                outcome = raw
            elif isinstance(raw, bool):
                outcome = ValidationOutcome(validation_id, raw)
            elif isinstance(raw, Mapping):
                outcome = ValidationOutcome(
                    validation_id=raw.get("validation_id", validation_id),
                    passed=raw.get("passed", False),
                    receipt_id=raw.get("receipt_id", ""),
                    reason_code=raw.get("reason_code", ""),
                )
            else:
                raise PolicyValidationError(
                    "fallback validation mapping values are invalid"
                )
            result[outcome.validation_id] = outcome
        return result
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        for raw in values:
            if not isinstance(raw, ValidationOutcome):
                raise PolicyValidationError(
                    "fallback validations must contain ValidationOutcome values"
                )
            result[raw.validation_id] = raw
        return result
    raise PolicyValidationError("fallback validations have an unsupported shape")


@dataclass(frozen=True)
class OverrideReceipt(CanonicalContract):
    """A bounded, expiring, content-addressed operator override."""

    SCHEMA: ClassVar[str] = OVERRIDE_RECEIPT_SCHEMA

    policy_id: str
    repository_tree_id: str
    paths: Tuple[str, ...]
    actor: str
    reason: str
    issued_at: str
    expires_at: str
    ast_scope_ids: Tuple[str, ...] = ()
    invariant_classes: Tuple[str, ...] = ()
    allowed_modes: Tuple[RolloutMode, ...] = (
        RolloutMode.CANARY,
        RolloutMode.ENFORCEMENT,
    )
    ticket_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", required=True)
        )
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, "repository_tree_id", required=True),
        )
        paths = tuple(
            sorted(
                {
                    _normalize_git_path(item, "paths")
                    for item in _strings(self.paths, "paths", required=True)
                }
            )
        )
        if any(any(char in path for char in "*?[") for path in paths):
            raise PolicyValidationError("override paths must be exact, not globs")
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "actor", _text(self.actor, "actor", required=True))
        object.__setattr__(self, "reason", _text(self.reason, "reason", required=True))
        issued_text, issued = _timestamp(self.issued_at, "issued_at")
        expires_text, expires = _timestamp(self.expires_at, "expires_at")
        if expires <= issued:
            raise PolicyValidationError("expires_at must be after issued_at")
        if (expires - issued).total_seconds() > MAX_OVERRIDE_LIFETIME_SECONDS:
            raise PolicyValidationError(
                "override lifetime exceeds the hard safety bound"
            )
        object.__setattr__(self, "issued_at", issued_text)
        object.__setattr__(self, "expires_at", expires_text)
        object.__setattr__(
            self, "ast_scope_ids", _strings(self.ast_scope_ids, "ast_scope_ids")
        )
        object.__setattr__(
            self,
            "invariant_classes",
            _strings(self.invariant_classes, "invariant_classes"),
        )
        modes = tuple(
            sorted(
                {
                    _enum(item, RolloutMode, "allowed_modes")
                    for item in self.allowed_modes
                },
                key=lambda item: item.rank,
            )
        )
        if not modes or any(not mode.blocks for mode in modes):
            raise PolicyValidationError(
                "override allowed_modes may only contain canary and enforcement"
            )
        object.__setattr__(self, "allowed_modes", modes)
        object.__setattr__(self, "ticket_id", _text(self.ticket_id, "ticket_id"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "repository_tree_id": self.repository_tree_id,
            "paths": self.paths,
            "ast_scope_ids": self.ast_scope_ids,
            "invariant_classes": self.invariant_classes,
            "allowed_modes": self.allowed_modes,
            "actor": self.actor,
            "reason": self.reason,
            "ticket_id": self.ticket_id,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
        }

    @classmethod
    def create(
        cls,
        *,
        policy_id: str,
        repository_tree_id: str,
        paths: Sequence[str],
        actor: str,
        reason: str,
        ttl_seconds: int,
        now: datetime | str | None = None,
        ast_scope_ids: Sequence[str] = (),
        invariant_classes: Sequence[str] = (),
        allowed_modes: Sequence[RolloutMode] = (
            RolloutMode.CANARY,
            RolloutMode.ENFORCEMENT,
        ),
        ticket_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "OverrideReceipt":
        ttl = _integer(
            ttl_seconds,
            "ttl_seconds",
            minimum=1,
            maximum=MAX_OVERRIDE_LIFETIME_SECONDS,
        )
        issued = _now(now)
        expires = issued + timedelta(seconds=ttl)
        return cls(
            policy_id=policy_id,
            repository_tree_id=repository_tree_id,
            paths=tuple(paths),
            actor=actor,
            reason=reason,
            issued_at=issued.isoformat().replace("+00:00", "Z"),
            expires_at=expires.isoformat().replace("+00:00", "Z"),
            ast_scope_ids=tuple(ast_scope_ids),
            invariant_classes=tuple(invariant_classes),
            allowed_modes=tuple(allowed_modes),
            ticket_id=ticket_id,
            metadata=metadata or {},
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OverrideReceipt":
        _schema(payload, cls.SCHEMA)
        result = cls(
            policy_id=payload.get("policy_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            paths=tuple(payload.get("paths") or ()),
            ast_scope_ids=tuple(payload.get("ast_scope_ids") or ()),
            invariant_classes=tuple(payload.get("invariant_classes") or ()),
            allowed_modes=tuple(
                payload.get("allowed_modes")
                or (RolloutMode.CANARY, RolloutMode.ENFORCEMENT)
            ),
            actor=payload.get("actor", ""),
            reason=payload.get("reason", ""),
            ticket_id=payload.get("ticket_id", ""),
            issued_at=payload.get("issued_at", ""),
            expires_at=payload.get("expires_at", ""),
            metadata=payload.get("metadata") or {},
        )
        claimed = payload.get("receipt_id") or payload.get("content_id")
        if claimed and claimed != result.receipt_id:
            raise PolicyValidationError("override receipt identity does not match")
        return result

    def covers(self, requirement: ProofRequirement, mode: RolloutMode) -> bool:
        """Return whether this exact receipt scope covers ``requirement``."""

        if mode not in self.allowed_modes or requirement.path not in self.paths:
            return False
        if self.ast_scope_ids and not set(requirement.ast_scope_ids).issubset(
            self.ast_scope_ids
        ):
            return False
        if self.invariant_classes and not set(requirement.invariant_classes).issubset(
            self.invariant_classes
        ):
            return False
        return True

    def invalid_reasons(
        self,
        *,
        policy: FormalVerificationPolicy,
        selection: PolicySelection,
        now: datetime | str | None = None,
    ) -> Tuple[str, ...]:
        reasons = []
        current = _now(now)
        issued = _timestamp(self.issued_at, "issued_at")[1]
        expires = _timestamp(self.expires_at, "expires_at")[1]
        if self.policy_id != policy.policy_id:
            reasons.append("override_policy_mismatch")
        if (
            not selection.repository_tree_id
            or self.repository_tree_id != selection.repository_tree_id
        ):
            reasons.append("override_tree_mismatch")
        if current < issued:
            reasons.append("override_not_yet_valid")
        if current >= expires:
            reasons.append("override_expired")
        if (expires - issued).total_seconds() > policy.max_override_seconds:
            reasons.append("override_exceeds_policy_duration")
        return tuple(reasons)

    def is_valid(
        self,
        *,
        policy: FormalVerificationPolicy,
        selection: PolicySelection,
        now: datetime | str | None = None,
    ) -> bool:
        return not self.invalid_reasons(policy=policy, selection=selection, now=now)


PolicyOverrideReceipt = ProofOverrideReceipt = OverrideReceipt


class OverrideReceiptStore:
    """Small append-only filesystem store for durable override receipts."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self.directory = Path(directory)

    def persist(self, receipt: OverrideReceipt) -> Path:
        if not isinstance(receipt, OverrideReceipt):
            raise PolicyValidationError("only OverrideReceipt values can be persisted")
        self.directory.mkdir(parents=True, exist_ok=True)
        destination = self.directory / f"{receipt.receipt_id}.json"
        payload = receipt.to_json().encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(destination, flags, 0o600)
        except FileExistsError:
            existing = self.load(receipt.receipt_id)
            if existing != receipt:
                raise PolicyValidationError(
                    "existing override receipt does not match its content identity"
                )
            return destination
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except Exception:
            try:
                destination.unlink()
            except OSError:
                pass
            raise
        return destination

    def load(self, receipt_id: str) -> OverrideReceipt:
        normalized = _text(receipt_id, "receipt_id", required=True)
        if "/" in normalized or "\\" in normalized or normalized in (".", ".."):
            raise PolicyValidationError("receipt_id is not a safe content identity")
        path = self.directory / f"{normalized}.json"
        if path.is_symlink():
            raise PolicyValidationError("override receipt must not be a symbolic link")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PolicyValidationError(
                "override receipt is unavailable or malformed"
            ) from exc
        if not isinstance(payload, Mapping):
            raise PolicyValidationError("override receipt must contain an object")
        receipt = OverrideReceipt.from_dict(payload)
        if receipt.receipt_id != normalized:
            raise PolicyValidationError("override receipt filename identity mismatch")
        return receipt


@dataclass(frozen=True)
class RolloutTransitionReceipt(CanonicalContract):
    """Durable authority and evidence for promotion or rollback."""

    SCHEMA: ClassVar[str] = ROLLOUT_TRANSITION_RECEIPT_SCHEMA

    policy_id: str
    from_mode: RolloutMode
    target_mode: RolloutMode
    actor: str
    reason: str
    issued_at: str
    observation_count: int = 0
    blocking_result_count: int = 0
    override_count: int = 0
    evidence_receipt_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_id", _text(self.policy_id, "policy_id", required=True)
        )
        object.__setattr__(
            self, "from_mode", _enum(self.from_mode, RolloutMode, "from_mode")
        )
        object.__setattr__(
            self, "target_mode", _enum(self.target_mode, RolloutMode, "target_mode")
        )
        if self.from_mode is self.target_mode:
            raise PolicyValidationError("rollout transition must change mode")
        if not (
            self.from_mode.can_promote_to(self.target_mode)
            or self.from_mode.can_roll_back_to(self.target_mode)
        ):
            raise PolicyValidationError(
                "promotion advances one stage; rollback targets an earlier stage"
            )
        object.__setattr__(self, "actor", _text(self.actor, "actor", required=True))
        object.__setattr__(self, "reason", _text(self.reason, "reason", required=True))
        issued, _ = _timestamp(self.issued_at, "issued_at")
        object.__setattr__(self, "issued_at", issued)
        for name in (
            "observation_count",
            "blocking_result_count",
            "override_count",
        ):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=0)
            )
        if self.blocking_result_count > self.observation_count:
            raise PolicyValidationError(
                "blocking_result_count cannot exceed observation_count"
            )
        if self.override_count > self.observation_count:
            raise PolicyValidationError(
                "override_count cannot exceed observation_count"
            )
        object.__setattr__(
            self,
            "evidence_receipt_ids",
            _strings(self.evidence_receipt_ids, "evidence_receipt_ids"),
        )

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "from_mode": self.from_mode,
            "target_mode": self.target_mode,
            "actor": self.actor,
            "reason": self.reason,
            "issued_at": self.issued_at,
            "observation_count": self.observation_count,
            "blocking_result_count": self.blocking_result_count,
            "override_count": self.override_count,
            "evidence_receipt_ids": self.evidence_receipt_ids,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RolloutTransitionReceipt":
        _schema(payload, cls.SCHEMA)
        result = cls(
            policy_id=payload.get("policy_id", ""),
            from_mode=payload.get("from_mode", RolloutMode.DISABLED),
            target_mode=payload.get("target_mode", RolloutMode.SHADOW),
            actor=payload.get("actor", ""),
            reason=payload.get("reason", ""),
            issued_at=payload.get("issued_at", ""),
            observation_count=payload.get("observation_count", 0),
            blocking_result_count=payload.get("blocking_result_count", 0),
            override_count=payload.get("override_count", 0),
            evidence_receipt_ids=tuple(payload.get("evidence_receipt_ids") or ()),
        )
        claimed = payload.get("receipt_id") or payload.get("content_id")
        if claimed and claimed != result.receipt_id:
            raise PolicyValidationError(
                "rollout transition receipt identity does not match"
            )
        return result


RolloutPromotionReceipt = PolicyTransitionReceipt = RolloutTransitionReceipt


@dataclass(frozen=True)
class RequirementGateResult(CanonicalContract):
    """Auditable result for one selected proof requirement."""

    SCHEMA: ClassVar[str] = REQUIREMENT_GATE_RESULT_SCHEMA

    requirement_id: str
    path: str
    effective_mode: RolloutMode
    proof_status: ProofResultStatus
    required_assurance: AssuranceLevel
    authoritative_assurance: AssuranceLevel
    proof_receipt_id: str
    proof_satisfied: bool
    fallback_satisfied: bool
    missing_or_failed_validations: Tuple[str, ...]
    requirement_satisfied: bool
    allowed: bool
    would_block: bool
    override_receipt_id: str = ""
    reason_codes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requirement_id",
            _text(self.requirement_id, "requirement_id", required=True),
        )
        object.__setattr__(self, "path", _normalize_git_path(self.path))
        object.__setattr__(
            self,
            "effective_mode",
            _enum(self.effective_mode, RolloutMode, "effective_mode"),
        )
        object.__setattr__(
            self,
            "proof_status",
            _enum(self.proof_status, ProofResultStatus, "proof_status"),
        )
        for name in ("required_assurance", "authoritative_assurance"):
            object.__setattr__(
                self, name, _enum(getattr(self, name), AssuranceLevel, name)
            )
        object.__setattr__(
            self,
            "proof_receipt_id",
            _text(self.proof_receipt_id, "proof_receipt_id"),
        )
        for name in (
            "proof_satisfied",
            "fallback_satisfied",
            "requirement_satisfied",
            "allowed",
            "would_block",
        ):
            if not isinstance(getattr(self, name), bool):
                raise PolicyValidationError(f"{name} must be a boolean")
        object.__setattr__(
            self,
            "missing_or_failed_validations",
            _strings(
                self.missing_or_failed_validations,
                "missing_or_failed_validations",
            ),
        )
        object.__setattr__(
            self,
            "override_receipt_id",
            _text(self.override_receipt_id, "override_receipt_id"),
        )
        object.__setattr__(
            self, "reason_codes", _strings(self.reason_codes, "reason_codes")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "path": self.path,
            "effective_mode": self.effective_mode,
            "proof_status": self.proof_status,
            "required_assurance": self.required_assurance,
            "authoritative_assurance": self.authoritative_assurance,
            "proof_receipt_id": self.proof_receipt_id,
            "proof_satisfied": self.proof_satisfied,
            "fallback_satisfied": self.fallback_satisfied,
            "missing_or_failed_validations": self.missing_or_failed_validations,
            "requirement_satisfied": self.requirement_satisfied,
            "allowed": self.allowed,
            "would_block": self.would_block,
            "override_receipt_id": self.override_receipt_id,
            "reason_codes": self.reason_codes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RequirementGateResult":
        _schema(payload, cls.SCHEMA)
        return cls(
            requirement_id=payload.get("requirement_id", ""),
            path=payload.get("path", ""),
            effective_mode=payload.get("effective_mode", RolloutMode.DISABLED),
            proof_status=payload.get("proof_status", ProofResultStatus.MISSING),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.UNVERIFIED
            ),
            authoritative_assurance=payload.get(
                "authoritative_assurance", AssuranceLevel.UNVERIFIED
            ),
            proof_receipt_id=payload.get("proof_receipt_id", ""),
            proof_satisfied=payload.get("proof_satisfied", False),
            fallback_satisfied=payload.get("fallback_satisfied", False),
            missing_or_failed_validations=tuple(
                payload.get("missing_or_failed_validations") or ()
            ),
            requirement_satisfied=payload.get("requirement_satisfied", False),
            allowed=payload.get("allowed", False),
            would_block=payload.get("would_block", False),
            override_receipt_id=payload.get("override_receipt_id", ""),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )


@dataclass(frozen=True)
class PolicyGateDecision(CanonicalContract):
    """Aggregate decision returned to a merge or completion gate."""

    SCHEMA: ClassVar[str] = POLICY_GATE_DECISION_SCHEMA

    policy_id: str
    selection_id: str
    repository_tree_id: str
    rollout_mode: RolloutMode
    allowed: bool
    results: Tuple[RequirementGateResult, ...]
    override_receipt_id: str = ""
    override_rejection_reasons: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("policy_id", "selection_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=True)
            )
        object.__setattr__(
            self,
            "repository_tree_id",
            _text(self.repository_tree_id, "repository_tree_id"),
        )
        object.__setattr__(
            self, "rollout_mode", _enum(self.rollout_mode, RolloutMode, "rollout_mode")
        )
        if not isinstance(self.allowed, bool):
            raise PolicyValidationError("allowed must be a boolean")
        results = tuple(sorted(self.results, key=lambda result: result.requirement_id))
        if not all(isinstance(item, RequirementGateResult) for item in results):
            raise PolicyValidationError(
                "results must contain RequirementGateResult values"
            )
        if self.allowed != all(item.allowed for item in results):
            raise PolicyValidationError("aggregate allowed result is inconsistent")
        object.__setattr__(self, "results", results)
        object.__setattr__(
            self,
            "override_receipt_id",
            _text(self.override_receipt_id, "override_receipt_id"),
        )
        object.__setattr__(
            self,
            "override_rejection_reasons",
            _strings(self.override_rejection_reasons, "override_rejection_reasons"),
        )

    @property
    def decision_id(self) -> str:
        return self.content_id

    @property
    def blocked_requirements(self) -> Tuple[str, ...]:
        return tuple(item.requirement_id for item in self.results if not item.allowed)

    @property
    def shadow_would_block(self) -> Tuple[str, ...]:
        return tuple(
            item.requirement_id
            for item in self.results
            if item.effective_mode is RolloutMode.SHADOW and item.would_block
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "selection_id": self.selection_id,
            "repository_tree_id": self.repository_tree_id,
            "rollout_mode": self.rollout_mode,
            "allowed": self.allowed,
            "results": self.results,
            "override_receipt_id": self.override_receipt_id,
            "override_rejection_reasons": self.override_rejection_reasons,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyGateDecision":
        _schema(payload, cls.SCHEMA)
        result = cls(
            policy_id=payload.get("policy_id", ""),
            selection_id=payload.get("selection_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            rollout_mode=payload.get("rollout_mode", RolloutMode.DISABLED),
            allowed=payload.get("allowed", False),
            results=tuple(
                (
                    item
                    if isinstance(item, RequirementGateResult)
                    else RequirementGateResult.from_dict(item)
                )
                for item in payload.get("results") or ()
            ),
            override_receipt_id=payload.get("override_receipt_id", ""),
            override_rejection_reasons=tuple(
                payload.get("override_rejection_reasons") or ()
            ),
        )
        claimed = payload.get("decision_id") or payload.get("content_id")
        if claimed and claimed != result.decision_id:
            raise PolicyValidationError("gate decision identity does not match")
        return result


GateDecision = ProofGateDecision = PolicyGateDecision


def _merge_status_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Normalize a bounded provider/cache status snapshot.

    Status is audit context, never proof authority.  Accepting a short string
    keeps adapters simple while serializing one unambiguous object shape.
    """

    if value is None:
        return {}
    if isinstance(value, (str, Enum)):
        text = value.value if isinstance(value, Enum) else value
        return {"status": _text(text, field_name, required=True)}
    return _mapping(value, field_name)


def _merge_proof_plan_snapshot(
    value: Any,
    claimed_plan_id: str = "",
) -> tuple[Mapping[str, Any], str]:
    """Return a canonical plan snapshot and independently checked identity."""

    if value is None:
        if claimed_plan_id:
            raise PolicyValidationError(
                "proof_plan_id cannot be supplied without a proof plan snapshot"
            )
        return {}, ""
    if isinstance(value, ProofPlan):
        plan = value
        snapshot = plan.to_dict()
        plan_id = plan.plan_id
    else:
        converter = getattr(value, "to_dict", None)
        if callable(converter):
            value = converter()
        snapshot = _mapping(value, "proof_plan")
        if not snapshot:
            if claimed_plan_id:
                raise PolicyValidationError(
                    "proof_plan_id cannot be supplied with an empty proof plan"
                )
            return {}, ""
        if snapshot.get("schema") == ProofPlan.SCHEMA:
            plan = ProofPlan.from_dict(snapshot)
            snapshot = plan.to_dict()
            plan_id = plan.plan_id
        else:
            plan_id = _text(
                snapshot.get("plan_id") or snapshot.get("content_id"),
                "proof_plan.plan_id",
                required=True,
            )
    supplied = _text(claimed_plan_id, "proof_plan_id")
    if supplied and supplied != plan_id:
        raise PolicyValidationError("proof plan identity does not match snapshot")
    return snapshot, plan_id


def _merge_outcomes(
    values: Any,
    requirements: Sequence[ProofRequirement],
) -> Tuple[ProofOutcome, ...]:
    """Normalize outcomes while rejecting duplicates and foreign requirements."""

    if values is None:
        return ()
    if isinstance(values, (ProofOutcome, ProofReceipt)):
        raw_values: Sequence[Any] = (values,)
    elif isinstance(values, Mapping):
        normalized = []
        for key, raw in values.items():
            requirement_id = str(key)
            if isinstance(raw, ProofOutcome):
                outcome = raw
            elif isinstance(raw, ProofReceipt):
                outcome = ProofOutcome.from_receipt(requirement_id, raw)
            elif isinstance(raw, Mapping):
                outcome = ProofOutcome(
                    requirement_id=raw.get("requirement_id", requirement_id),
                    status=raw.get("status", ProofResultStatus.MISSING),
                    authoritative_assurance=raw.get(
                        "authoritative_assurance",
                        raw.get("assurance", AssuranceLevel.UNVERIFIED),
                    ),
                    receipt_id=raw.get("receipt_id", ""),
                    reason_code=raw.get("reason_code", ""),
                )
            else:
                raise PolicyValidationError(
                    "proof outcome mapping values are invalid"
                )
            normalized.append(outcome)
        raw_values = normalized
    elif isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        raw_values = values
    else:
        raise PolicyValidationError("proof outcomes have an unsupported shape")

    result = []
    for raw in raw_values:
        if isinstance(raw, ProofReceipt):
            if len(requirements) != 1:
                raise PolicyValidationError(
                    "a single proof receipt can only evaluate one requirement"
                )
            outcome = ProofOutcome.from_receipt(
                requirements[0].requirement_id, raw
            )
        elif isinstance(raw, ProofOutcome):
            outcome = raw
        elif isinstance(raw, Mapping):
            outcome = ProofOutcome.from_dict(raw)
        else:
            raise PolicyValidationError(
                "proof outcomes must contain ProofOutcome values"
            )
        result.append(outcome)

    requirement_ids = {item.requirement_id for item in requirements}
    outcome_ids = [item.requirement_id for item in result]
    if len(set(outcome_ids)) != len(outcome_ids):
        raise PolicyValidationError("proof outcomes contain duplicate requirements")
    foreign = sorted(set(outcome_ids) - requirement_ids)
    if foreign:
        raise PolicyValidationError(
            "proof outcome references an unselected requirement: "
            + ", ".join(foreign)
        )
    return tuple(sorted(result, key=lambda item: item.requirement_id))


def _merge_validations(values: Any) -> Tuple[ValidationOutcome, ...]:
    if values is None:
        return ()
    normalized = _validation_map(values)
    if isinstance(values, Sequence) and not isinstance(
        values, (str, bytes, bytearray)
    ):
        supplied_ids = [
            item.validation_id
            if isinstance(item, ValidationOutcome)
            else str(
                item.get("validation_id", "")
                if isinstance(item, Mapping)
                else ""
            )
            for item in values
        ]
        supplied_ids = [item for item in supplied_ids if item]
        if len(set(supplied_ids)) != len(supplied_ids):
            raise PolicyValidationError(
                "fallback validations contain duplicate validation IDs"
            )
    return tuple(sorted(normalized.values(), key=lambda item: item.validation_id))


@dataclass(frozen=True)
class MergeProofGateReceipt(CanonicalContract):
    """Canonical proof-policy decision at the merge-promotion boundary.

    This receipt embeds every snapshot needed to reproduce the decision.  It
    does not trust summary fields from providers or queue metadata: typed proof
    receipts are revalidated, identities are cross-checked, and the policy
    decision must be for the exact candidate tree and selection.
    """

    SCHEMA: ClassVar[str] = MERGE_PROOF_GATE_RECEIPT_SCHEMA

    policy: FormalVerificationPolicy
    selection: PolicySelection
    repository_tree_id: str
    proof_plan: Mapping[str, Any]
    proof_plan_id: str
    proof_outcomes: Tuple[ProofOutcome, ...]
    validation_outcomes: Tuple[ValidationOutcome, ...]
    proof_receipts: Tuple[ProofReceipt, ...]
    proof_receipt_ids: Tuple[str, ...]
    validation_receipt_ids: Tuple[str, ...]
    decision: PolicyGateDecision
    evaluated_at: str
    override: OverrideReceipt | None = None
    provider_status: Mapping[str, Any] = field(default_factory=dict)
    provider_error: str = ""
    cache_status: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policy = (
            self.policy
            if isinstance(self.policy, FormalVerificationPolicy)
            else FormalVerificationPolicy.from_dict(self.policy)
        )
        selection = (
            self.selection
            if isinstance(self.selection, PolicySelection)
            else PolicySelection.from_dict(self.selection)
        )
        decision = (
            self.decision
            if isinstance(self.decision, PolicyGateDecision)
            else PolicyGateDecision.from_dict(self.decision)
        )
        override = self.override
        if override is not None and not isinstance(override, OverrideReceipt):
            override = OverrideReceipt.from_dict(override)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "override", override)

        tree_id = _text(
            self.repository_tree_id, "repository_tree_id", required=True
        )
        object.__setattr__(self, "repository_tree_id", tree_id)
        plan_snapshot, plan_id = _merge_proof_plan_snapshot(
            self.proof_plan, self.proof_plan_id
        )
        object.__setattr__(self, "proof_plan", plan_snapshot)
        object.__setattr__(self, "proof_plan_id", plan_id)

        outcomes = _merge_outcomes(self.proof_outcomes, selection.requirements)
        validations = _merge_validations(self.validation_outcomes)
        object.__setattr__(self, "proof_outcomes", outcomes)
        object.__setattr__(self, "validation_outcomes", validations)

        receipts = tuple(
            sorted(
                [
                    item
                    if isinstance(item, ProofReceipt)
                    else ProofReceipt.from_dict(item)
                    for item in self.proof_receipts
                ],
                key=lambda item: item.receipt_id,
            )
        )
        if len({item.receipt_id for item in receipts}) != len(receipts):
            raise PolicyValidationError("proof receipts contain duplicate identities")
        if receipts and not plan_snapshot:
            raise PolicyValidationError(
                "typed proof receipts require an exact proof plan snapshot"
            )
        object.__setattr__(self, "proof_receipts", receipts)
        receipt_ids = _strings(self.proof_receipt_ids, "proof_receipt_ids")
        snapshot_ids = tuple(item.receipt_id for item in receipts)
        if not set(snapshot_ids).issubset(receipt_ids):
            raise PolicyValidationError(
                "every proof receipt snapshot must be represented by proof_receipt_ids"
            )
        object.__setattr__(self, "proof_receipt_ids", receipt_ids)
        validation_receipt_ids = _strings(
            self.validation_receipt_ids, "validation_receipt_ids"
        )
        represented_validation_ids = {
            item.receipt_id for item in validations if item.receipt_id
        }
        if not represented_validation_ids.issubset(validation_receipt_ids):
            raise PolicyValidationError(
                "every validation receipt must be represented by "
                "validation_receipt_ids"
            )
        object.__setattr__(
            self, "validation_receipt_ids", validation_receipt_ids
        )
        evaluated_text, _ = _timestamp(self.evaluated_at, "evaluated_at")
        object.__setattr__(self, "evaluated_at", evaluated_text)
        object.__setattr__(
            self,
            "provider_status",
            _merge_status_mapping(self.provider_status, "provider_status"),
        )
        object.__setattr__(
            self, "provider_error", _text(self.provider_error, "provider_error")
        )
        object.__setattr__(
            self,
            "cache_status",
            _merge_status_mapping(self.cache_status, "cache_status"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

        if selection.policy_id != policy.policy_id:
            raise PolicyValidationError("selection policy does not match receipt policy")
        if selection.repository_tree_id != tree_id:
            raise PolicyValidationError("selection tree does not match merge tree")
        if decision.policy_id != policy.policy_id:
            raise PolicyValidationError("decision policy does not match receipt policy")
        if decision.selection_id != selection.selection_id:
            raise PolicyValidationError(
                "decision selection does not match receipt selection"
            )
        if decision.repository_tree_id != tree_id:
            raise PolicyValidationError("decision tree does not match merge tree")
        if decision.rollout_mode is not policy.rollout_mode:
            raise PolicyValidationError(
                "decision rollout mode does not match receipt policy"
            )
        expected_requirements = {
            item.requirement_id for item in selection.requirements
        }
        decision_requirements = {
            item.requirement_id for item in decision.results
        }
        if decision_requirements != expected_requirements:
            raise PolicyValidationError(
                "decision results do not exactly cover selected requirements"
            )
        requirements_by_id = {
            item.requirement_id: item for item in selection.requirements
        }
        validations_by_id = {
            item.validation_id: item for item in validations
        }
        for result in decision.results:
            if not result.fallback_satisfied:
                continue
            requirement = requirements_by_id[result.requirement_id]
            undurable = [
                validation_id
                for validation_id in requirement.fallback_validations
                if validation_id not in validations_by_id
                or not validations_by_id[validation_id].passed
                or not validations_by_id[validation_id].receipt_id
            ]
            if undurable:
                raise PolicyValidationError(
                    "merge fallback requires durable passing validation receipts: "
                    + ", ".join(sorted(undurable))
                )

        if plan_snapshot:
            plan_policy = plan_snapshot.get("policy_id")
            plan_tree = plan_snapshot.get("repository_tree_id") or plan_snapshot.get(
                "candidate_tree_id"
            )
            if plan_policy and str(plan_policy) != policy.policy_id:
                raise PolicyValidationError("proof plan policy does not match")
            if plan_tree and str(plan_tree) != tree_id:
                raise PolicyValidationError("proof plan tree does not match")
            plan_obligations = {
                str(item)
                for item in plan_snapshot.get("obligation_ids", ()) or ()
                if str(item)
            }
            if plan_obligations:
                foreign_obligations = sorted(
                    {
                        item.obligation_id
                        for item in receipts
                        if item.obligation_id not in plan_obligations
                    }
                )
                if foreign_obligations:
                    raise PolicyValidationError(
                        "proof receipt obligation is outside the proof plan: "
                        + ", ".join(foreign_obligations)
                    )

        snapshots_by_id = {item.receipt_id: item for item in receipts}
        for receipt in receipts:
            if receipt.policy_id != policy.policy_id:
                raise PolicyValidationError("proof receipt policy does not match")
            if receipt.repository_tree_id != tree_id:
                raise PolicyValidationError("proof receipt tree does not match")
            if plan_id and receipt.plan_id != plan_id:
                raise PolicyValidationError("proof receipt plan does not match")
        used_receipt_ids = {
            item.receipt_id for item in outcomes if item.receipt_id
        } | {
            item.proof_receipt_id
            for item in decision.results
            if item.proof_receipt_id
        }
        if not used_receipt_ids.issubset(receipt_ids):
            raise PolicyValidationError(
                "used proof receipt IDs are not represented in the merge receipt"
            )
        proved_outcomes_without_snapshot = [
            item.requirement_id
            for item in outcomes
            if item.status is ProofResultStatus.PROVED
            and (
                not item.receipt_id
                or item.receipt_id not in snapshots_by_id
            )
        ]
        proof_results_without_snapshot = [
            item.requirement_id
            for item in decision.results
            if item.proof_satisfied
            and (
                not item.proof_receipt_id
                or item.proof_receipt_id not in snapshots_by_id
            )
        ]
        if proved_outcomes_without_snapshot or proof_results_without_snapshot:
            missing = sorted(
                set(
                    proved_outcomes_without_snapshot
                    + proof_results_without_snapshot
                )
            )
            raise PolicyValidationError(
                "proved merge outcomes require embedded typed proof receipts: "
                + ", ".join(missing)
            )
        for outcome in outcomes:
            receipt = snapshots_by_id.get(outcome.receipt_id)
            if receipt is None:
                continue
            projected = ProofOutcome.from_receipt(outcome.requirement_id, receipt)
            if (
                outcome.status is not projected.status
                or outcome.authoritative_assurance
                is not projected.authoritative_assurance
                or outcome.receipt_id != projected.receipt_id
            ):
                raise PolicyValidationError(
                    "proof outcome does not match its typed proof receipt"
                )

        if override is not None:
            if override.policy_id != policy.policy_id:
                raise PolicyValidationError("override policy does not match")
            if override.repository_tree_id != tree_id:
                raise PolicyValidationError("override tree does not match")
        if decision.override_receipt_id:
            if override is None or decision.override_receipt_id != override.receipt_id:
                raise PolicyValidationError(
                    "decision override identity is not represented"
                )
        reproduced_decision = policy.evaluate_gate(
            selection,
            outcomes,
            validations=validations,
            override=override,
            now=self.evaluated_at,
        )
        if reproduced_decision != decision:
            raise PolicyValidationError(
                "merge proof gate decision is not reproducible from embedded evidence"
            )

    @property
    def policy_id(self) -> str:
        return self.policy.policy_id

    @property
    def selection_id(self) -> str:
        return self.selection.selection_id

    @property
    def decision_id(self) -> str:
        return self.decision.decision_id

    @property
    def override_receipt_id(self) -> str:
        return self.override.receipt_id if self.override is not None else ""

    @property
    def allowed(self) -> bool:
        return self.decision.allowed

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy": self.policy,
            "selection_id": self.selection_id,
            "selection": self.selection,
            "repository_tree_id": self.repository_tree_id,
            "proof_plan_id": self.proof_plan_id,
            "proof_plan": self.proof_plan,
            "proof_outcomes": self.proof_outcomes,
            "validation_outcomes": self.validation_outcomes,
            "proof_receipt_ids": self.proof_receipt_ids,
            "proof_receipts": self.proof_receipts,
            "validation_receipt_ids": self.validation_receipt_ids,
            "decision_id": self.decision_id,
            "decision": self.decision,
            "allowed": self.allowed,
            "override_receipt_id": self.override_receipt_id,
            "override": self.override,
            "provider_status": self.provider_status,
            "provider_error": self.provider_error,
            "cache_status": self.cache_status,
            "evaluated_at": self.evaluated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def build(
        cls,
        *,
        policy: FormalVerificationPolicy,
        selection: PolicySelection,
        repository_tree_id: str,
        proof_plan: Any = None,
        outcomes: Any = None,
        validations: Any = None,
        proof_receipts: Sequence[ProofReceipt | Mapping[str, Any]] = (),
        proof_receipt_ids: Sequence[str] = (),
        override: OverrideReceipt | None = None,
        provider_status: Any = None,
        provider_error: str = "",
        cache_status: Any = None,
        now: datetime | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "MergeProofGateReceipt":
        """Evaluate ``policy`` and bind the resulting exact merge evidence."""

        normalized_policy = (
            policy
            if isinstance(policy, FormalVerificationPolicy)
            else FormalVerificationPolicy.from_dict(policy)
        )
        normalized_selection = (
            selection
            if isinstance(selection, PolicySelection)
            else PolicySelection.from_dict(selection)
        )
        normalized_receipts = tuple(
            item
            if isinstance(item, ProofReceipt)
            else ProofReceipt.from_dict(item)
            for item in proof_receipts
        )
        normalized_outcomes = _merge_outcomes(
            outcomes, normalized_selection.requirements
        )
        if outcomes is None and normalized_receipts:
            derived = []
            unused = list(normalized_receipts)
            for requirement in normalized_selection.requirements:
                matching = [
                    item
                    for item in unused
                    if item.metadata.get("requirement_id")
                    == requirement.requirement_id
                ]
                if not matching and len(normalized_selection.requirements) == 1:
                    matching = list(unused)
                if len(matching) == 1:
                    receipt = matching[0]
                    derived.append(
                        ProofOutcome.from_receipt(
                            requirement.requirement_id, receipt
                        )
                    )
                    unused.remove(receipt)
            normalized_outcomes = tuple(derived)
        normalized_validations = _merge_validations(validations)
        normalized_override = override
        if normalized_override is not None and not isinstance(
            normalized_override, OverrideReceipt
        ):
            if not isinstance(normalized_override, Mapping):
                raise PolicyValidationError(
                    "override must be an OverrideReceipt or mapping"
                )
            normalized_override = OverrideReceipt.from_dict(normalized_override)
        evaluation_time = _now(now)
        decision = normalized_policy.evaluate_gate(
            normalized_selection,
            normalized_outcomes,
            validations=normalized_validations,
            override=normalized_override,
            now=evaluation_time,
        )
        plan_snapshot, plan_id = _merge_proof_plan_snapshot(proof_plan)
        all_receipt_ids = tuple(
            sorted(
                {
                    *(
                        _text(item, "proof_receipt_ids", required=True)
                        for item in proof_receipt_ids
                    ),
                    *(item.receipt_id for item in normalized_receipts),
                }
            )
        )
        validation_receipt_ids = tuple(
            sorted(
                {
                    item.receipt_id
                    for item in normalized_validations
                    if item.receipt_id
                }
            )
        )
        evaluated = evaluation_time.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )
        return cls(
            policy=normalized_policy,
            selection=normalized_selection,
            repository_tree_id=repository_tree_id,
            proof_plan=plan_snapshot,
            proof_plan_id=plan_id,
            proof_outcomes=normalized_outcomes,
            validation_outcomes=normalized_validations,
            proof_receipts=normalized_receipts,
            proof_receipt_ids=all_receipt_ids,
            validation_receipt_ids=validation_receipt_ids,
            decision=decision,
            override=normalized_override,
            provider_status=_merge_status_mapping(
                provider_status, "provider_status"
            ),
            provider_error=provider_error,
            cache_status=_merge_status_mapping(cache_status, "cache_status"),
            evaluated_at=evaluated,
            metadata=metadata or {},
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MergeProofGateReceipt":
        _schema(payload, cls.SCHEMA)
        result = cls(
            policy=FormalVerificationPolicy.from_dict(payload.get("policy") or {}),
            selection=PolicySelection.from_dict(payload.get("selection") or {}),
            repository_tree_id=payload.get("repository_tree_id", ""),
            proof_plan=payload.get("proof_plan") or {},
            proof_plan_id=payload.get("proof_plan_id", ""),
            proof_outcomes=tuple(
                ProofOutcome.from_dict(item)
                for item in payload.get("proof_outcomes", payload.get("outcomes", ()))
                or ()
            ),
            validation_outcomes=tuple(
                ValidationOutcome.from_dict(item)
                for item in payload.get(
                    "validation_outcomes", payload.get("validations", ())
                )
                or ()
            ),
            proof_receipts=tuple(
                ProofReceipt.from_dict(item)
                for item in payload.get("proof_receipts") or ()
            ),
            proof_receipt_ids=tuple(payload.get("proof_receipt_ids") or ()),
            validation_receipt_ids=tuple(
                payload.get("validation_receipt_ids") or ()
            ),
            decision=PolicyGateDecision.from_dict(payload.get("decision") or {}),
            override=(
                OverrideReceipt.from_dict(payload["override"])
                if isinstance(payload.get("override"), Mapping)
                else None
            ),
            provider_status=payload.get("provider_status") or {},
            provider_error=payload.get("provider_error", ""),
            cache_status=payload.get("cache_status") or {},
            evaluated_at=payload.get("evaluated_at", ""),
            metadata=payload.get("metadata") or {},
        )
        for name, actual in (
            ("policy_id", result.policy_id),
            ("selection_id", result.selection_id),
            ("decision_id", result.decision_id),
            ("override_receipt_id", result.override_receipt_id),
        ):
            claimed = payload.get(name)
            if claimed is not None and str(claimed) != actual:
                raise PolicyValidationError(
                    f"merge proof gate {name} does not match snapshot"
                )
        claimed_allowed = payload.get("allowed")
        if claimed_allowed is not None and claimed_allowed is not result.allowed:
            raise PolicyValidationError(
                "merge proof gate allowed value does not match decision"
            )
        claimed = payload.get("receipt_id") or payload.get("content_id")
        if claimed and claimed != result.receipt_id:
            raise PolicyValidationError(
                "merge proof gate receipt identity does not match"
            )
        return result


def build_merge_proof_gate_receipt(
    *,
    policy: FormalVerificationPolicy,
    selection: PolicySelection,
    repository_tree_id: str,
    **kwargs: Any,
) -> MergeProofGateReceipt:
    """Functional spelling of :meth:`MergeProofGateReceipt.build`."""

    return MergeProofGateReceipt.build(
        policy=policy,
        selection=selection,
        repository_tree_id=repository_tree_id,
        **kwargs,
    )


def select_proof_requirements(
    policy: FormalVerificationPolicy,
    changes: Iterable[ChangedScope | Mapping[str, Any]],
    *,
    repository_tree_id: str = "",
) -> PolicySelection:
    """Functional spelling of :meth:`FormalVerificationPolicy.select`."""

    return policy.select(changes, repository_tree_id=repository_tree_id)


def evaluate_proof_gate(
    policy: FormalVerificationPolicy,
    selection: PolicySelection,
    outcomes: Any = None,
    *,
    validations: Any = None,
    override: OverrideReceipt | None = None,
    now: datetime | str | None = None,
) -> PolicyGateDecision:
    """Functional spelling of :meth:`FormalVerificationPolicy.evaluate_gate`."""

    return policy.evaluate_gate(
        selection,
        outcomes,
        validations=validations,
        override=override,
        now=now,
    )


def default_formal_verification_policy(
    mode: RolloutMode = RolloutMode.SHADOW,
) -> FormalVerificationPolicy:
    """Return the reviewed baseline policy for supervisor safety invariants.

    Unrelated paths and unclassified changes do not match these rules.  The
    baseline intentionally does not configure a generic fallback for critical
    invariants; deployments can add a reviewed, named validation rule where a
    sound fallback exists.
    """

    return FormalVerificationPolicy(
        name="agent-supervisor-safety",
        version="1",
        rollout_mode=mode,
        rules=(
            ProofPolicyRule(
                rule_id="critical-supervisor-invariants",
                path_patterns=(
                    "ipfs_datasets_py/ipfs_accelerate_py/"
                    "ipfs_accelerate_py/agent_supervisor/**",
                ),
                minimum_risk=RiskLevel.CRITICAL,
                invariant_classes=tuple(item.value for item in InvariantClass),
                required_assurance=AssuranceLevel.KERNEL_VERIFIED,
                description="Critical modeled supervisor invariants require a kernel.",
            ),
            ProofPolicyRule(
                rule_id="high-risk-modeled-invariants",
                minimum_risk=RiskLevel.HIGH,
                invariant_classes=tuple(item.value for item in InvariantClass),
                required_assurance=AssuranceLevel.SOLVER_CHECKED,
                description="High-risk modeled invariants require independent solving.",
            ),
        ),
        canary_percent=0,
        canary_salt="agent-supervisor-safety-v1",
    )


default_policy = default_formal_verification_policy


__all__ = [
    "CHANGED_SCOPE_SCHEMA",
    "DEFAULT_MAX_OVERRIDE_SECONDS",
    "FORMAL_VERIFICATION_POLICY_SCHEMA",
    "MAX_OVERRIDE_LIFETIME_SECONDS",
    "MERGE_PROOF_GATE_RECEIPT_SCHEMA",
    "OVERRIDE_RECEIPT_SCHEMA",
    "POLICY_GATE_DECISION_SCHEMA",
    "POLICY_SELECTION_SCHEMA",
    "POLICY_VERSION",
    "PROOF_OUTCOME_SCHEMA",
    "PROOF_POLICY_RULE_SCHEMA",
    "PROOF_REQUIREMENT_SCHEMA",
    "REQUIREMENT_GATE_RESULT_SCHEMA",
    "ROLLOUT_TRANSITION_RECEIPT_SCHEMA",
    "SCHEMA_VERSION",
    "VALIDATION_OUTCOME_SCHEMA",
    "ChangeScope",
    "ChangedCodeScope",
    "ChangedScope",
    "FallbackValidationResult",
    "FormalVerificationPolicy",
    "FormalVerificationRule",
    "GateDecision",
    "InvariantClass",
    "MergeProofGateReceipt",
    "OverrideReceipt",
    "OverrideReceiptStore",
    "PolicyGateDecision",
    "PolicyOverrideReceipt",
    "PolicyRule",
    "PolicySelection",
    "PolicyTransitionReceipt",
    "PolicyValidationError",
    "ProofGateDecision",
    "ProofGateOutcome",
    "ProofOutcome",
    "ProofOverrideReceipt",
    "ProofPolicyRule",
    "ProofRequirement",
    "ProofResult",
    "ProofResultStatus",
    "RequirementGateResult",
    "RiskLevel",
    "RiskSelectedProofPolicy",
    "RolloutMode",
    "RolloutPromotionReceipt",
    "RolloutTransitionReceipt",
    "ValidationOutcome",
    "ValidationResult",
    "VerificationPolicy",
    "default_formal_verification_policy",
    "default_policy",
    "build_merge_proof_gate_receipt",
    "evaluate_proof_gate",
    "select_proof_requirements",
]
