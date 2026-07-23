"""Bounded, fail-closed routing for unsuccessful proof obligations.

Proof failures are useful implementation evidence, but raw solver output is
not safe task context and fallback tests are not formal assurance.  This
module provides the boundary between those concerns:

* counterexamples and unsat cores become small canonical diagnostics and
  regression fixtures;
* only obligation-declared validation IDs are routed;
* named validation IDs become executable only through a trusted command
  catalog; and
* rollout mode controls progress without changing the underlying proof
  verdict or authoritative assurance.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .formal_verification_contracts import (
    AssuranceLevel,
    CanonicalContract,
    CodeProofObligation,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)
from .formal_verification_policy import (
    ProofOutcome,
    ProofResultStatus,
    RolloutMode,
)
from .validation_commands import (
    DeclaredValidation,
    ValidationCommand,
    ValidationRequirementKind,
    build_declared_validations,
)


PROOF_FALLBACK_VERSION = 1
PROOF_DIAGNOSTIC_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-fallback-diagnostic@1"
)
REGRESSION_FIXTURE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-regression-fixture@1"
)
PROOF_FALLBACK_PLAN_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/proof-fallback-plan@1"
)

DEFAULT_MAX_DIAGNOSTIC_BYTES = 8 * 1024
DEFAULT_MAX_FIXTURE_BYTES = 16 * 1024
DEFAULT_MAX_TEXT_CHARS = 1024
DEFAULT_MAX_COLLECTION_ITEMS = 32
DEFAULT_MAX_NESTING_DEPTH = 5
DEFAULT_MAX_DIAGNOSTICS = 16
_DIAGNOSTIC_PAYLOAD_BYTES = DEFAULT_MAX_DIAGNOSTIC_BYTES // 2

_SENSITIVE_KEY_RE = re.compile(
    r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|session[_-]?token|credential|authorization|cookie|"
    r"private[_-]?key|hidden[_-]?witness)",
    re.IGNORECASE,
)
_COUNTEREXAMPLE_KEYS = (
    "counterexample",
    "counter_example",
    "counterexample_trace",
    "attack_trace",
    "hypertrace",
    "model",
    "trace",
    "assignment",
    "witness",
)
_UNSAT_CORE_KEYS = ("unsat_core", "unsat-core", "unsatisfiable_core", "core")


class ProofFallbackValidationError(ContractValidationError):
    """Raised when fallback evidence cannot be routed safely."""


class ProofFailureKind(str, Enum):
    COUNTEREXAMPLE = "counterexample"
    UNSAT_CORE = "unsat_core"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


class RegressionExpectation(str, Enum):
    REJECT = "reject"
    UNSATISFIABLE = "unsatisfiable"


def _enum(value: Any, kind: type[Enum], field_name: str) -> Any:
    if isinstance(value, kind):
        return value
    raw = getattr(value, "value", value)
    try:
        return kind(str(raw).strip().lower())
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in kind)
        raise ProofFallbackValidationError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ProofFallbackValidationError(
            f"unsupported schema {supplied!r}; expected {expected}"
        )


def _text(
    value: Any,
    field_name: str,
    *,
    required: bool = False,
    maximum: int = DEFAULT_MAX_TEXT_CHARS,
) -> str:
    if value is None:
        result = ""
    elif isinstance(value, str):
        result = value.strip()
    else:
        result = str(value).strip()
    if len(result) > maximum:
        result = result[: max(0, maximum - 1)] + "…"
    if required and not result:
        raise ProofFallbackValidationError(f"{field_name} is required")
    return result


@dataclass
class _NormalizationBudget:
    remaining_items: int
    maximum_text: int
    maximum_depth: int
    truncated: bool = False
    redacted: bool = False


def _bounded_value(value: Any, budget: _NormalizationBudget, depth: int = 0) -> Any:
    """Return a deterministic JSON value under structural and text bounds."""

    if budget.remaining_items <= 0:
        budget.truncated = True
        return "<truncated>"
    budget.remaining_items -= 1
    if depth >= budget.maximum_depth:
        budget.truncated = True
        return "<maximum-depth>"
    if value is None or isinstance(value, bool) or (
        isinstance(value, int) and not isinstance(value, bool)
    ):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "<non-finite-number>"
        # Canonical proof contracts reject floats.  A decimal string retains
        # the diagnostic value without weakening the canonical boundary.
        return format(value, ".17g")
    if isinstance(value, str):
        if len(value) > budget.maximum_text:
            budget.truncated = True
            return value[: max(0, budget.maximum_text - 1)] + "…"
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        budget.redacted = True
        return f"<binary-redacted:{len(value)}-bytes>"
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        keys = sorted(str(key) for key in value)
        if len(keys) > DEFAULT_MAX_COLLECTION_ITEMS:
            keys = keys[:DEFAULT_MAX_COLLECTION_ITEMS]
            budget.truncated = True
        for key in keys:
            if _SENSITIVE_KEY_RE.search(key):
                result[key] = "<redacted>"
                budget.redacted = True
                continue
            # Look up the original key without accepting non-string-key
            # ambiguity.  String keys are the normal provider contract.
            original_key = next((raw for raw in value if str(raw) == key), key)
            result[key] = _bounded_value(
                value[original_key], budget, depth + 1
            )
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        items = list(value)
        if len(items) > DEFAULT_MAX_COLLECTION_ITEMS:
            items = items[:DEFAULT_MAX_COLLECTION_ITEMS]
            budget.truncated = True
        return [_bounded_value(item, budget, depth + 1) for item in items]
    if isinstance(value, (set, frozenset)):
        items = sorted(value, key=lambda item: repr(item))
        if len(items) > DEFAULT_MAX_COLLECTION_ITEMS:
            items = items[:DEFAULT_MAX_COLLECTION_ITEMS]
            budget.truncated = True
        return sorted(
            (_bounded_value(item, budget, depth + 1) for item in items),
            key=lambda item: canonical_json_bytes(item),
        )
    budget.truncated = True
    return f"<unsupported-type:{type(value).__name__}>"


def _bounded_payload(
    value: Any,
    *,
    maximum_bytes: int,
    maximum_items: int = DEFAULT_MAX_COLLECTION_ITEMS,
    maximum_text: int = DEFAULT_MAX_TEXT_CHARS,
    maximum_depth: int = DEFAULT_MAX_NESTING_DEPTH,
) -> tuple[Any, bool, bool]:
    if maximum_bytes < 256:
        raise ProofFallbackValidationError("maximum_bytes must be at least 256")
    budget = _NormalizationBudget(
        remaining_items=max(1, int(maximum_items)),
        maximum_text=max(16, int(maximum_text)),
        maximum_depth=max(1, int(maximum_depth)),
    )
    normalized = _bounded_value(value, budget)
    encoded = canonical_json_bytes(normalized)
    if len(encoded) > maximum_bytes:
        digest = hashlib.sha256(encoded).hexdigest()
        normalized = {
            "digest": f"sha256:{digest}",
            "omitted": "<payload-exceeded-byte-limit>",
        }
        budget.truncated = True
    return normalized, budget.truncated, budget.redacted


def normalize_counterexample(
    value: Any,
    *,
    maximum_bytes: int = DEFAULT_MAX_DIAGNOSTIC_BYTES,
) -> tuple[Any, str, bool, bool]:
    """Normalize one counterexample and return payload, identity and flags."""

    payload, truncated, redacted = _bounded_payload(
        value, maximum_bytes=maximum_bytes
    )
    identity = content_identity(
        {"kind": ProofFailureKind.COUNTEREXAMPLE.value, "payload": payload}
    )
    return payload, identity, truncated, redacted


def normalize_unsat_core(
    value: Any,
    *,
    maximum_bytes: int = DEFAULT_MAX_DIAGNOSTIC_BYTES,
) -> tuple[Any, str, bool, bool]:
    """Normalize an unsat core with set-like ordering semantics."""

    payload, truncated, redacted = _bounded_payload(
        value, maximum_bytes=maximum_bytes
    )
    if isinstance(payload, list):
        by_value = {canonical_json_bytes(item): item for item in payload}
        payload = [by_value[key] for key in sorted(by_value)]
    identity = content_identity(
        {"kind": ProofFailureKind.UNSAT_CORE.value, "payload": payload}
    )
    return payload, identity, truncated, redacted


@dataclass(frozen=True)
class ProofFallbackDiagnostic(CanonicalContract):
    """Compact task-facing representation of one proof failure."""

    SCHEMA = PROOF_DIAGNOSTIC_SCHEMA

    obligation_id: str
    repository_tree_id: str
    task_id: str
    kind: ProofFailureKind
    counterexample_id: str
    summary: str
    payload: Any = field(default_factory=dict)
    source_id: str = ""
    truncated: bool = False
    redacted: bool = False

    def __post_init__(self) -> None:
        for name in ("obligation_id", "repository_tree_id", "counterexample_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=True)
            )
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, ProofFailureKind, "kind")
        )
        object.__setattr__(
            self,
            "summary",
            _text(self.summary, "summary", required=True),
        )
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id"))
        if not isinstance(self.truncated, bool) or not isinstance(self.redacted, bool):
            raise ProofFallbackValidationError(
                "truncated and redacted must be booleans"
            )
        payload, bounded, redacted = _bounded_payload(
            self.payload, maximum_bytes=_DIAGNOSTIC_PAYLOAD_BYTES
        )
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "truncated", self.truncated or bounded)
        object.__setattr__(self, "redacted", self.redacted or redacted)

    @property
    def diagnostic_id(self) -> str:
        return self.content_id

    @property
    def deduplication_key(self) -> str:
        """Identity required by REF-256: obligation + tree + example."""

        return content_identity(
            {
                "obligation_id": self.obligation_id,
                "repository_tree_id": self.repository_tree_id,
                "counterexample_id": self.counterexample_id,
            }
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "fallback_version": PROOF_FALLBACK_VERSION,
            "obligation_id": self.obligation_id,
            "repository_tree_id": self.repository_tree_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "counterexample_id": self.counterexample_id,
            "summary": self.summary,
            "payload": self.payload,
            "source_id": self.source_id,
            "truncated": self.truncated,
            "redacted": self.redacted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofFallbackDiagnostic":
        _schema(payload, cls.SCHEMA)
        result = cls(
            obligation_id=payload.get("obligation_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            task_id=payload.get("task_id", ""),
            kind=payload.get("kind", ProofFailureKind.ERROR),
            counterexample_id=payload.get("counterexample_id", ""),
            summary=payload.get("summary", ""),
            payload=payload.get("payload"),
            source_id=payload.get("source_id", ""),
            truncated=payload.get("truncated", False),
            redacted=payload.get("redacted", False),
        )
        claimed = payload.get("diagnostic_id") or payload.get("content_id")
        if claimed and claimed != result.diagnostic_id:
            raise ProofFallbackValidationError(
                "diagnostic content identity does not match"
            )
        return result


@dataclass(frozen=True)
class ProofRegressionFixture(CanonicalContract):
    """A bounded regression seed derived from a diagnostic."""

    SCHEMA = REGRESSION_FIXTURE_SCHEMA

    obligation_id: str
    repository_tree_id: str
    task_id: str
    counterexample_id: str
    kind: ProofFailureKind
    input_data: Any
    expected: RegressionExpectation
    diagnostic_id: str

    def __post_init__(self) -> None:
        for name in (
            "obligation_id",
            "repository_tree_id",
            "counterexample_id",
            "diagnostic_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=True)
            )
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(
            self, "kind", _enum(self.kind, ProofFailureKind, "kind")
        )
        object.__setattr__(
            self,
            "expected",
            _enum(self.expected, RegressionExpectation, "expected"),
        )
        bounded, _truncated, _redacted = _bounded_payload(
            self.input_data, maximum_bytes=DEFAULT_MAX_FIXTURE_BYTES
        )
        object.__setattr__(self, "input_data", bounded)

    @property
    def fixture_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "fallback_version": PROOF_FALLBACK_VERSION,
            "obligation_id": self.obligation_id,
            "repository_tree_id": self.repository_tree_id,
            "task_id": self.task_id,
            "counterexample_id": self.counterexample_id,
            "kind": self.kind,
            "input_data": self.input_data,
            "expected": self.expected,
            "diagnostic_id": self.diagnostic_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofRegressionFixture":
        _schema(payload, cls.SCHEMA)
        result = cls(
            obligation_id=payload.get("obligation_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            task_id=payload.get("task_id", ""),
            counterexample_id=payload.get("counterexample_id", ""),
            kind=payload.get("kind", ProofFailureKind.COUNTEREXAMPLE),
            input_data=payload.get("input_data"),
            expected=payload.get("expected", RegressionExpectation.REJECT),
            diagnostic_id=payload.get("diagnostic_id", ""),
        )
        claimed = payload.get("fixture_id") or payload.get("content_id")
        if claimed and claimed != result.fixture_id:
            raise ProofFallbackValidationError(
                "fixture content identity does not match"
            )
        return result


def build_regression_fixture(
    diagnostic: ProofFallbackDiagnostic,
) -> ProofRegressionFixture | None:
    """Build a fixture for semantic counterexamples and unsat cores."""

    if diagnostic.kind not in {
        ProofFailureKind.COUNTEREXAMPLE,
        ProofFailureKind.UNSAT_CORE,
    }:
        return None
    return ProofRegressionFixture(
        obligation_id=diagnostic.obligation_id,
        repository_tree_id=diagnostic.repository_tree_id,
        task_id=diagnostic.task_id,
        counterexample_id=diagnostic.counterexample_id,
        kind=diagnostic.kind,
        input_data=diagnostic.payload,
        expected=(
            RegressionExpectation.REJECT
            if diagnostic.kind is ProofFailureKind.COUNTEREXAMPLE
            else RegressionExpectation.UNSATISFIABLE
        ),
        diagnostic_id=diagnostic.diagnostic_id,
    )


class ProofFallbackDeduplicator:
    """Thread-safe identity index for repeated equivalent proof failures."""

    def __init__(self, identities: Iterable[str] = ()) -> None:
        self._lock = threading.Lock()
        self._identities = set(str(item) for item in identities if str(item))

    def register(self, diagnostic: ProofFallbackDiagnostic) -> bool:
        """Return ``True`` exactly once for a deduplication identity."""

        if not isinstance(diagnostic, ProofFallbackDiagnostic):
            raise ProofFallbackValidationError(
                "deduplication requires a ProofFallbackDiagnostic"
            )
        key = diagnostic.deduplication_key
        with self._lock:
            if key in self._identities:
                return False
            self._identities.add(key)
            return True

    def contains(self, diagnostic: ProofFallbackDiagnostic) -> bool:
        with self._lock:
            return diagnostic.deduplication_key in self._identities

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._identities)

    def snapshot(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._identities))

    def clear(self) -> None:
        with self._lock:
            self._identities.clear()


@dataclass(frozen=True)
class ProofFallbackPlan(CanonicalContract):
    """Focused validation and rollout disposition for one failed obligation."""

    SCHEMA = PROOF_FALLBACK_PLAN_SCHEMA

    obligation_id: str
    repository_tree_id: str
    task_id: str
    proof_status: ProofResultStatus
    rollout_mode: RolloutMode
    required_assurance: AssuranceLevel
    diagnostics: tuple[ProofFallbackDiagnostic, ...] = ()
    regression_fixtures: tuple[ProofRegressionFixture, ...] = ()
    validations: tuple[DeclaredValidation, ...] = ()
    duplicate_diagnostic_ids: tuple[str, ...] = ()
    can_continue: bool = False
    blocking: bool = True
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("obligation_id", "repository_tree_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=True)
            )
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(
            self,
            "proof_status",
            _enum(self.proof_status, ProofResultStatus, "proof_status"),
        )
        object.__setattr__(
            self, "rollout_mode", _enum(self.rollout_mode, RolloutMode, "rollout_mode")
        )
        object.__setattr__(
            self,
            "required_assurance",
            _enum(self.required_assurance, AssuranceLevel, "required_assurance"),
        )
        if any(
            not isinstance(item, ProofFallbackDiagnostic)
            for item in self.diagnostics
        ):
            raise ProofFallbackValidationError(
                "diagnostics must contain ProofFallbackDiagnostic values"
            )
        if any(
            not isinstance(item, ProofRegressionFixture)
            for item in self.regression_fixtures
        ):
            raise ProofFallbackValidationError(
                "regression_fixtures must contain ProofRegressionFixture values"
            )
        if any(not isinstance(item, DeclaredValidation) for item in self.validations):
            raise ProofFallbackValidationError(
                "validations must contain DeclaredValidation values"
            )
        object.__setattr__(
            self,
            "duplicate_diagnostic_ids",
            tuple(sorted(set(self.duplicate_diagnostic_ids))),
        )
        object.__setattr__(
            self, "reason_codes", tuple(dict.fromkeys(self.reason_codes))
        )
        if not isinstance(self.can_continue, bool) or not isinstance(
            self.blocking, bool
        ):
            raise ProofFallbackValidationError(
                "can_continue and blocking must be booleans"
            )
        if self.can_continue == self.blocking:
            raise ProofFallbackValidationError(
                "can_continue must be the inverse of blocking"
            )

    @property
    def plan_id(self) -> str:
        return self.content_id

    @property
    def executable_commands(self) -> tuple[ValidationCommand, ...]:
        return tuple(
            item.command for item in self.validations if item.command is not None
        )

    @property
    def manual_review_requirements(self) -> tuple[DeclaredValidation, ...]:
        return tuple(
            item
            for item in self.validations
            if item.kind is ValidationRequirementKind.MANUAL_REVIEW
        )

    @property
    def unresolved_validation_ids(self) -> tuple[str, ...]:
        return tuple(
            item.validation_id
            for item in self.validations
            if item.command is None
            and item.kind is not ValidationRequirementKind.MANUAL_REVIEW
        )

    @property
    def deduplicated_count(self) -> int:
        return len(self.duplicate_diagnostic_ids)

    def to_proof_outcome(self, requirement_id: str | None = None) -> ProofOutcome:
        """Project the unchanged failed verdict into the policy gate contract."""

        return ProofOutcome(
            requirement_id=_text(
                requirement_id or self.obligation_id,
                "requirement_id",
                required=True,
            ),
            status=self.proof_status,
            authoritative_assurance=AssuranceLevel.UNVERIFIED,
            reason_code=(
                self.reason_codes[0] if self.reason_codes else "proof_fallback"
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "fallback_version": PROOF_FALLBACK_VERSION,
            "obligation_id": self.obligation_id,
            "repository_tree_id": self.repository_tree_id,
            "task_id": self.task_id,
            "proof_status": self.proof_status,
            "rollout_mode": self.rollout_mode,
            "required_assurance": self.required_assurance,
            "diagnostics": tuple(item.to_dict() for item in self.diagnostics),
            "regression_fixtures": tuple(
                item.to_dict() for item in self.regression_fixtures
            ),
            "validations": tuple(item.to_dict() for item in self.validations),
            "duplicate_diagnostic_ids": self.duplicate_diagnostic_ids,
            "can_continue": self.can_continue,
            "blocking": self.blocking,
            "reason_codes": self.reason_codes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofFallbackPlan":
        _schema(payload, cls.SCHEMA)
        diagnostics = tuple(
            item
            if isinstance(item, ProofFallbackDiagnostic)
            else ProofFallbackDiagnostic.from_dict(item)
            for item in payload.get("diagnostics") or ()
        )
        fixtures = tuple(
            item
            if isinstance(item, ProofRegressionFixture)
            else ProofRegressionFixture.from_dict(item)
            for item in payload.get("regression_fixtures") or ()
        )
        validations = tuple(
            item
            if isinstance(item, DeclaredValidation)
            else DeclaredValidation.from_dict(item)
            for item in payload.get("validations") or ()
        )
        result = cls(
            obligation_id=payload.get("obligation_id", ""),
            repository_tree_id=payload.get("repository_tree_id", ""),
            task_id=payload.get("task_id", ""),
            proof_status=payload.get(
                "proof_status", ProofResultStatus.INCONCLUSIVE
            ),
            rollout_mode=payload.get("rollout_mode", RolloutMode.SHADOW),
            required_assurance=payload.get(
                "required_assurance", AssuranceLevel.KERNEL_VERIFIED
            ),
            diagnostics=diagnostics,
            regression_fixtures=fixtures,
            validations=validations,
            duplicate_diagnostic_ids=tuple(
                payload.get("duplicate_diagnostic_ids") or ()
            ),
            can_continue=payload.get("can_continue", False),
            blocking=payload.get("blocking", True),
            reason_codes=tuple(payload.get("reason_codes") or ()),
        )
        claimed = payload.get("plan_id") or payload.get("content_id")
        if claimed and claimed != result.plan_id:
            raise ProofFallbackValidationError(
                "fallback plan content identity does not match"
            )
        return result


def _mapping_from_result(result: Any) -> Mapping[str, Any]:
    if result is None:
        return {}
    if isinstance(result, Mapping):
        return result
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        return value if isinstance(value, Mapping) else {}
    return {}


def _status(value: Any, result: Any) -> ProofResultStatus:
    raw = value
    if raw is None:
        mapping = _mapping_from_result(result)
        raw = mapping.get("status", mapping.get("verdict", mapping.get("outcome")))
    raw = getattr(raw, "value", raw)
    aliases = {
        "timeout": ProofResultStatus.TIMED_OUT,
        "timed_out": ProofResultStatus.TIMED_OUT,
        "unknown": ProofResultStatus.INCONCLUSIVE,
        "malformed": ProofResultStatus.ERROR,
        "blocked": ProofResultStatus.INCONCLUSIVE,
    }
    if str(raw or "").strip().lower() in aliases:
        return aliases[str(raw).strip().lower()]
    return _enum(raw or ProofResultStatus.INCONCLUSIVE, ProofResultStatus, "status")


def _obligation_fields(
    obligation: CodeProofObligation | Mapping[str, Any],
) -> tuple[str, str, str, AssuranceLevel, tuple[str, ...]]:
    if isinstance(obligation, CodeProofObligation):
        return (
            obligation.obligation_id,
            obligation.repository_tree_id,
            obligation.task_id,
            obligation.required_assurance,
            obligation.fallback_checks,
        )
    if not isinstance(obligation, Mapping):
        raise ProofFallbackValidationError(
            "obligation must be a CodeProofObligation or mapping"
        )
    obligation_id = _text(
        obligation.get("obligation_id")
        or obligation.get("content_id")
        or obligation.get("id"),
        "obligation_id",
        required=True,
    )
    tree_id = _text(
        obligation.get("repository_tree_id")
        or obligation.get("tree_id")
        or obligation.get("candidate_tree_id"),
        "repository_tree_id",
        required=True,
    )
    checks = obligation.get("fallback_checks") or obligation.get(
        "fallback_validations"
    ) or ()
    if isinstance(checks, str):
        checks = (checks,)
    if not isinstance(checks, Sequence):
        raise ProofFallbackValidationError("fallback_checks must be a sequence")
    return (
        obligation_id,
        tree_id,
        _text(obligation.get("task_id"), "task_id"),
        _enum(
            obligation.get("required_assurance", AssuranceLevel.KERNEL_VERIFIED),
            AssuranceLevel,
            "required_assurance",
        ),
        tuple(_text(item, "fallback_check", required=True) for item in checks),
    )


def _find_values(value: Any, keys: tuple[str, ...], *, depth: int = 0) -> list[Any]:
    """Find explicit evidence fields without retaining surrounding transcripts."""

    if depth > DEFAULT_MAX_NESTING_DEPTH:
        return []
    result: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in keys:
                result.append(item)
            elif normalized_key in {
                "evidence",
                "metadata",
                "diagnostics",
                "attempts",
                "result",
                "output",
            }:
                result.extend(_find_values(item, keys, depth=depth + 1))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in list(value)[:DEFAULT_MAX_COLLECTION_ITEMS]:
            result.extend(_find_values(item, keys, depth=depth + 1))
    return result[:DEFAULT_MAX_DIAGNOSTICS]


def _failure_diagnostic(
    *,
    obligation_id: str,
    repository_tree_id: str,
    task_id: str,
    kind: ProofFailureKind,
    value: Any,
    source_id: str,
) -> ProofFallbackDiagnostic:
    if kind is ProofFailureKind.COUNTEREXAMPLE:
        payload, identity, truncated, redacted = normalize_counterexample(value)
        summary = "The obligation has a bounded counterexample; add a regression that rejects it."
    elif kind is ProofFailureKind.UNSAT_CORE:
        payload, identity, truncated, redacted = normalize_unsat_core(value)
        summary = "The selected assumptions contain a bounded unsatisfiable core."
    else:
        payload, truncated, redacted = _bounded_payload(
            value, maximum_bytes=DEFAULT_MAX_DIAGNOSTIC_BYTES
        )
        identity = content_identity({"kind": kind.value, "payload": payload})
        summary = {
            ProofFailureKind.UNSUPPORTED: (
                "No reviewed proof route supports this obligation; run only its "
                "declared fallback validation."
            ),
            ProofFailureKind.INCONCLUSIVE: (
                "Proof execution was inconclusive; focused validation is still required."
            ),
            ProofFailureKind.ERROR: (
                "Proof execution failed without authoritative assurance."
            ),
        }[kind]
    return ProofFallbackDiagnostic(
        obligation_id=obligation_id,
        repository_tree_id=repository_tree_id,
        task_id=task_id,
        kind=kind,
        counterexample_id=identity,
        summary=summary,
        payload=payload,
        source_id=source_id,
        truncated=truncated,
        redacted=redacted,
    )


class ProofFallbackRouter:
    """Stateful router whose identity index deduplicates across calls."""

    def __init__(
        self,
        *,
        command_catalog: Mapping[str, str | ValidationCommand] | None = None,
        deduplicator: ProofFallbackDeduplicator | None = None,
    ) -> None:
        self.command_catalog = dict(command_catalog or {})
        self.deduplicator = deduplicator or ProofFallbackDeduplicator()

    def route(
        self,
        obligation: CodeProofObligation | Mapping[str, Any],
        result: Any = None,
        *,
        status: ProofResultStatus | str | None = None,
        counterexample: Any = None,
        unsat_core: Any = None,
        rollout_mode: RolloutMode | str = RolloutMode.SHADOW,
        fallback_checks: Iterable[str] | None = None,
        source_id: str = "",
    ) -> ProofFallbackPlan:
        (
            obligation_id,
            repository_tree_id,
            task_id,
            required_assurance,
            declared_checks,
        ) = _obligation_fields(obligation)
        proof_status = _status(status, result)
        if proof_status is ProofResultStatus.PROVED:
            raise ProofFallbackValidationError(
                "a proved obligation must not be routed through proof fallbacks"
            )
        mode = _enum(rollout_mode, RolloutMode, "rollout_mode")
        result_mapping = _mapping_from_result(result)

        counterexamples = (
            [counterexample]
            if counterexample is not None
            else _find_values(result_mapping, _COUNTEREXAMPLE_KEYS)
        )
        cores = (
            [unsat_core]
            if unsat_core is not None
            else _find_values(result_mapping, _UNSAT_CORE_KEYS)
        )
        candidates: list[ProofFallbackDiagnostic] = []
        for value in counterexamples:
            candidates.append(
                _failure_diagnostic(
                    obligation_id=obligation_id,
                    repository_tree_id=repository_tree_id,
                    task_id=task_id,
                    kind=ProofFailureKind.COUNTEREXAMPLE,
                    value=value,
                    source_id=source_id,
                )
            )
        for value in cores:
            candidates.append(
                _failure_diagnostic(
                    obligation_id=obligation_id,
                    repository_tree_id=repository_tree_id,
                    task_id=task_id,
                    kind=ProofFailureKind.UNSAT_CORE,
                    value=value,
                    source_id=source_id,
                )
            )

        if not candidates:
            kind = {
                ProofResultStatus.UNSUPPORTED: ProofFailureKind.UNSUPPORTED,
                ProofResultStatus.ERROR: ProofFailureKind.ERROR,
                ProofResultStatus.DISPROVED: ProofFailureKind.ERROR,
            }.get(proof_status, ProofFailureKind.INCONCLUSIVE)
            reason = result_mapping.get(
                "reason",
                result_mapping.get(
                    "reason_code",
                    result_mapping.get("detail", proof_status.value),
                ),
            )
            candidates.append(
                _failure_diagnostic(
                    obligation_id=obligation_id,
                    repository_tree_id=repository_tree_id,
                    task_id=task_id,
                    kind=kind,
                    value={"proof_status": proof_status.value, "reason": reason},
                    source_id=source_id,
                )
            )

        unique: list[ProofFallbackDiagnostic] = []
        duplicates: list[str] = []
        for diagnostic in candidates[:DEFAULT_MAX_DIAGNOSTICS]:
            if self.deduplicator.register(diagnostic):
                unique.append(diagnostic)
            else:
                duplicates.append(diagnostic.diagnostic_id)
        fixtures = tuple(
            fixture
            for fixture in (build_regression_fixture(item) for item in unique)
            if fixture is not None
        )

        checks = tuple(fallback_checks) if fallback_checks is not None else declared_checks
        if not checks:
            checks = ("manual:unsupported-proof-obligation",)
        validations = build_declared_validations(
            checks, command_catalog=self.command_catalog
        )

        if mode in {RolloutMode.DISABLED, RolloutMode.SHADOW}:
            can_continue = True
            blocking = False
            mode_reason = (
                "proof_policy_disabled"
                if mode is RolloutMode.DISABLED
                else "shadow_fallback_validation_continues"
            )
        else:
            # A fallback plan contains no proof receipt and therefore cannot
            # satisfy solver/kernel assurance.  The formal policy gate may
            # separately accept an explicitly allowed validation outcome, but
            # this routing layer never grants that permission itself.
            can_continue = False
            blocking = True
            mode_reason = "required_assurance_not_satisfied"

        reasons = [f"proof_{proof_status.value}", mode_reason]
        if any(item.manual_review_required for item in validations):
            reasons.append("manual_review_required")
        if any(
            not item.executable and not item.manual_review_required
            for item in validations
        ):
            reasons.append("declared_validation_unresolved")
        if duplicates:
            reasons.append("equivalent_failure_deduplicated")
        return ProofFallbackPlan(
            obligation_id=obligation_id,
            repository_tree_id=repository_tree_id,
            task_id=task_id,
            proof_status=proof_status,
            rollout_mode=mode,
            required_assurance=required_assurance,
            diagnostics=tuple(unique),
            regression_fixtures=fixtures,
            validations=validations,
            duplicate_diagnostic_ids=tuple(duplicates),
            can_continue=can_continue,
            blocking=blocking,
            reason_codes=tuple(reasons),
        )


def route_proof_fallback(
    obligation: CodeProofObligation | Mapping[str, Any],
    result: Any = None,
    **kwargs: Any,
) -> ProofFallbackPlan:
    """Route one failed proof using a fresh in-call deduplication scope."""

    command_catalog = kwargs.pop("command_catalog", None)
    deduplicator = kwargs.pop("deduplicator", None)
    return ProofFallbackRouter(
        command_catalog=command_catalog,
        deduplicator=deduplicator,
    ).route(obligation, result, **kwargs)


def route_proof_fallbacks(
    failures: Iterable[
        tuple[CodeProofObligation | Mapping[str, Any], Any]
        | CodeProofObligation
        | Mapping[str, Any]
    ],
    *,
    rollout_mode: RolloutMode | str = RolloutMode.SHADOW,
    command_catalog: Mapping[str, str | ValidationCommand] | None = None,
) -> tuple[ProofFallbackPlan, ...]:
    """Route several failures with one shared semantic deduplication index."""

    router = ProofFallbackRouter(command_catalog=command_catalog)
    plans: list[ProofFallbackPlan] = []
    for failure in failures:
        if (
            isinstance(failure, tuple)
            and len(failure) == 2
            and isinstance(failure[0], (CodeProofObligation, Mapping))
        ):
            obligation, result = failure
        elif isinstance(failure, CodeProofObligation):
            obligation, result = failure, None
        elif isinstance(failure, Mapping) and "obligation" in failure:
            obligation = failure["obligation"]
            result = failure.get("result", failure)
        else:
            obligation, result = failure, None
        plans.append(
            router.route(
                obligation, result, rollout_mode=rollout_mode  # type: ignore[arg-type]
            )
        )
    return tuple(plans)


# Compatibility-friendly semantic spellings.
CounterexampleDiagnostic = TaskDiagnostic = ProofFallbackDiagnostic
RegressionFixture = ProofRegressionFixture
FallbackPlan = ProofFallbackPlan
FallbackRouter = ProofFallbackRouter
route_proof_failure = route_proof_fallback


__all__ = [
    "DEFAULT_MAX_COLLECTION_ITEMS",
    "DEFAULT_MAX_DIAGNOSTIC_BYTES",
    "DEFAULT_MAX_DIAGNOSTICS",
    "DEFAULT_MAX_FIXTURE_BYTES",
    "DEFAULT_MAX_NESTING_DEPTH",
    "DEFAULT_MAX_TEXT_CHARS",
    "PROOF_DIAGNOSTIC_SCHEMA",
    "PROOF_FALLBACK_PLAN_SCHEMA",
    "PROOF_FALLBACK_VERSION",
    "REGRESSION_FIXTURE_SCHEMA",
    "CounterexampleDiagnostic",
    "FallbackPlan",
    "FallbackRouter",
    "ProofFailureKind",
    "ProofFallbackDeduplicator",
    "ProofFallbackDiagnostic",
    "ProofFallbackPlan",
    "ProofFallbackRouter",
    "ProofFallbackValidationError",
    "ProofRegressionFixture",
    "RegressionExpectation",
    "RegressionFixture",
    "TaskDiagnostic",
    "build_regression_fixture",
    "normalize_counterexample",
    "normalize_unsat_core",
    "route_proof_failure",
    "route_proof_fallback",
    "route_proof_fallbacks",
]
