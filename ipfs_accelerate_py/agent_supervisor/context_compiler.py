"""Provider-aware, token-budgeted supervisor context compilation.

The compiler in this module deliberately separates the non-truncatable
context core (goal, authority, scope, and acceptance) from ranked evidence.
It negotiates the usable input limit with the effective provider, emits a
decision for every evidence item, and represents omitted material with
bounded expansion references instead of copying source bodies into receipts.

Retry contexts are parent-bound deltas.  A delta may contain only changed or
explicitly requested evidence; :func:`reconstruct_context` deterministically
rebuilds the effective context and verifies that the immutable core, tree,
policy, and required evidence coverage were not weakened.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final

from .context_contracts import (
    ContextBudget,
    ContextCapsule,
    ContextContractError,
    ContextDeltaCapsule,
    ContextReference,
    ContextTier,
    canonical_context_json_bytes,
)
from .formal_verification_contracts import CanonicalContract


REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID: Final = (
    "208290439421789408250562066350459701853"
)
DELTA_RETRY_EVIDENCE_ID: Final = (
    "306437607356117177048620815571362227127"
)
CONTEXT_EVIDENCE_PRODUCERS: Final = {
    REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID: "context_compiler",
    DELTA_RETRY_EVIDENCE_ID: "context_delta_compiler",
}

CONTEXT_COMPILATION_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/context-compilation-receipt@1"
)
CONTEXT_DELTA_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/context-delta-receipt@1"
)
REQUIRED_CONTEXT_BUDGET_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "required-context-budget-evidence@1"
)
DELTA_RETRY_CONTEXT_EVIDENCE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/delta-retry-context-evidence@1"
)
CONTEXT_COMPILER_VERSION = 1
MAX_DECISIONS = 4_096
MAX_CALIBRATION_SAMPLES = 128
MAX_ERROR_BPS = 1_000_000


class ContextCompilationError(ContextContractError):
    """Base error raised when a safe context cannot be compiled."""


class RequiredContextOverflowError(ContextCompilationError):
    """The invariant core or explicitly required evidence does not fit."""


class ContextDeltaError(ContextCompilationError):
    """A retry delta is stale, lossy, unchanged, or not token efficient."""


class InclusionReason(str, Enum):
    """Why an evidence reference was included."""

    REQUIRED = "required"
    RANKED_FIT = "ranked_fit"
    CHANGED = "changed"
    REQUESTED = "requested"


class ExclusionReason(str, Enum):
    """Why an evidence reference was not transmitted."""

    TOKEN_BUDGET = "token_budget"
    ITEM_LIMIT = "item_limit"
    UNCHANGED = "unchanged"
    NOT_REQUESTED = "not_requested"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ContextCompilationError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise ContextCompilationError(f"{name} must not be empty")
    if "\x00" in result or len(result.encode("utf-8")) > 8_192:
        raise ContextCompilationError(f"{name} is not bounded text")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContextCompilationError(
            f"{name} must be an integer of at least {minimum}"
        )
    return value


def _strings(
    value: Iterable[Any],
    name: str,
    *,
    maximum: int = MAX_DECISIONS,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise ContextCompilationError(f"{name} must be a sequence")
    result: set[str] = set()
    for index, item in enumerate(value):
        if index >= maximum:
            raise ContextCompilationError(f"{name} exceeds its item limit")
        result.add(_text(item, name))
    return tuple(sorted(result))


def _digest(value: Any, name: str) -> str:
    result = _text(value, name)
    raw = result.removeprefix("sha256:")
    if len(raw) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in raw):
        raise ContextCompilationError(f"{name} must be a SHA-256 digest")
    return "sha256:" + raw.lower()


def _canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_context_json_bytes(value)).hexdigest()


def _reject_unknown(
    payload: Mapping[str, Any], allowed: set[str], noun: str
) -> None:
    if not isinstance(payload, Mapping):
        raise ContextCompilationError(f"{noun} must be an object")
    if set(payload).difference(allowed):
        raise ContextCompilationError(
            f"{noun} contains unsupported fields; rebuild its canonical payload"
        )


def _schema(payload: Mapping[str, Any], expected: str, noun: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ContextCompilationError(
            f"unsupported {noun} schema; rebuild the canonical payload"
        )
    version = payload.get("contract_version")
    if version not in (None, CONTEXT_COMPILER_VERSION):
        raise ContextCompilationError(
            f"unsupported {noun} contract version"
        )


def _check_identity(
    payload: Mapping[str, Any], actual: str, noun: str
) -> None:
    claimed = payload.get("content_id") or payload.get("receipt_id")
    if claimed not in (None, "", actual):
        raise ContextCompilationError(f"{noun} identity does not match payload")


def _coerce_references(
    value: Iterable[ContextReference | Mapping[str, Any]],
) -> tuple[ContextReference, ...]:
    result: dict[str, ContextReference] = {}
    for index, raw in enumerate(value):
        if index >= MAX_DECISIONS:
            raise ContextCompilationError(
                "evidence exceeds its reference-count limit"
            )
        item = (
            raw
            if isinstance(raw, ContextReference)
            else ContextReference.from_dict(raw)
            if isinstance(raw, Mapping)
            else None
        )
        if item is None:
            raise ContextCompilationError("evidence contains an invalid reference")
        if item.tier is ContextTier.EXPANSION:
            raise ContextCompilationError(
                "candidate evidence cannot use the expansion tier"
            )
        previous = result.get(item.reference_id)
        if previous is not None and previous != item:
            raise ContextCompilationError(
                "evidence contains conflicting duplicate reference IDs"
            )
        result[item.reference_id] = item
    return tuple(result[key] for key in sorted(result))


def _as_expansion(reference: ContextReference) -> ContextReference:
    return ContextReference(
        reference_id=reference.reference_id,
        kind=reference.kind,
        tier=ContextTier.EXPANSION,
        referenced_content_id=reference.referenced_content_id,
        repository_id=reference.repository_id,
        tree_id=reference.tree_id,
        path=reference.path,
        summary=reference.summary,
        byte_count=reference.byte_count,
        token_count=reference.token_count,
        metadata=reference.metadata,
    )


@dataclass(frozen=True)
class EvidenceSelectionDecision:
    """Deterministic selection audit entry for one evidence reference."""

    reference_id: str
    included: bool
    reason: InclusionReason | ExclusionReason | str
    token_count: int
    priority: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reference_id", _text(self.reference_id, "reference_id")
        )
        if not isinstance(self.included, bool):
            raise ContextCompilationError("included must be a boolean")
        enum_type = InclusionReason if self.included else ExclusionReason
        try:
            reason = (
                self.reason
                if isinstance(self.reason, enum_type)
                else enum_type(str(getattr(self.reason, "value", self.reason)))
            )
        except ValueError as exc:
            raise ContextCompilationError(
                "selection reason does not match inclusion state"
            ) from exc
        object.__setattr__(self, "reason", reason)
        object.__setattr__(
            self, "token_count", _integer(self.token_count, "token_count")
        )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ContextCompilationError("priority must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_id": self.reference_id,
            "included": self.included,
            "reason": self.reason.value,
            "token_count": self.token_count,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "EvidenceSelectionDecision":
        _reject_unknown(
            payload,
            {"reference_id", "included", "reason", "token_count", "priority"},
            "selection decision",
        )
        return cls(
            reference_id=payload.get("reference_id", ""),
            included=payload.get("included", False),
            reason=payload.get("reason", ""),
            token_count=payload.get("token_count", 0),
            priority=payload.get("priority", 0),
        )


class CalibratedTokenEstimator:
    """Use a provider tokenizer when available, otherwise a calibrated fallback."""

    def __init__(
        self,
        tokenizer: Callable[[str], Any] | Any | None = None,
        *,
        chars_per_token: int = 4,
    ) -> None:
        self._tokenizer = tokenizer
        self._chars_per_token = _integer(
            chars_per_token, "chars_per_token", minimum=1
        )
        self._samples: list[tuple[int, int]] = []

    @property
    def provider_aware(self) -> bool:
        return self._tokenizer is not None

    @property
    def name(self) -> str:
        return "provider_tokenizer" if self.provider_aware else "calibrated_utf8"

    @property
    def calibration_samples(self) -> int:
        return len(self._samples)

    @property
    def error_bps(self) -> int:
        if self.provider_aware:
            return 0
        if not self._samples:
            return 10_000
        total_actual = sum(actual for _, actual in self._samples)
        if total_actual == 0:
            return 0
        absolute_error = sum(
            abs(estimated - actual) for estimated, actual in self._samples
        )
        return min(MAX_ERROR_BPS, absolute_error * 10_000 // total_actual)

    def _provider_count(self, text: str) -> int:
        tokenizer = self._tokenizer
        if callable(tokenizer):
            result = tokenizer(text)
        elif hasattr(tokenizer, "encode"):
            result = tokenizer.encode(text)
        else:
            raise ContextCompilationError(
                "tokenizer must be callable or expose encode()"
            )
        if isinstance(result, bool):
            raise ContextCompilationError("provider tokenizer returned a boolean")
        if isinstance(result, int):
            return _integer(result, "provider token count")
        try:
            return len(result)
        except TypeError as exc:
            raise ContextCompilationError(
                "provider tokenizer returned an uncountable value"
            ) from exc

    def estimate(self, value: str | bytes | Any) -> int:
        if isinstance(value, bytes):
            text = value.decode("utf-8")
        elif isinstance(value, str):
            text = value
        else:
            text = canonical_context_json_bytes(value).decode("utf-8")
        if self.provider_aware:
            return self._provider_count(text)
        byte_count = len(text.encode("utf-8"))
        raw = max(1, (byte_count + self._chars_per_token - 1) // self._chars_per_token)
        if not self._samples:
            return raw
        estimated_total = sum(estimated for estimated, _ in self._samples)
        actual_total = sum(actual for _, actual in self._samples)
        if estimated_total == 0:
            return raw
        return max(1, (raw * actual_total + estimated_total - 1) // estimated_total)

    count = estimate

    def calibrate(self, value: str | bytes | Any, actual_tokens: int) -> None:
        actual = _integer(actual_tokens, "actual_tokens")
        if self.provider_aware:
            return
        if isinstance(value, bytes):
            byte_count = len(value)
        elif isinstance(value, str):
            byte_count = len(value.encode("utf-8"))
        else:
            byte_count = len(canonical_context_json_bytes(value))
        estimated = max(
            1, (byte_count + self._chars_per_token - 1) // self._chars_per_token
        )
        self._samples.append((estimated, actual))
        del self._samples[:-MAX_CALIBRATION_SAMPLES]


def _reference_tokens(
    estimator: CalibratedTokenEstimator, reference: ContextReference
) -> int:
    # A producer-supplied token count is a useful conservative hint, never an
    # authority boundary.  Always tokenize the canonical descriptor as well
    # so a large reference cannot claim ``token_count=1`` and escape the
    # effective provider budget.
    return max(
        reference.token_count,
        estimator.estimate(reference.to_record()),
    )


def _core_payload(
    *,
    goal: Any,
    authority: Any,
    scope: Any,
    acceptance: Any,
) -> dict[str, Any]:
    return {
        "goal": goal,
        "authority": authority,
        "scope": scope,
        "acceptance": acceptance,
    }


def context_provider_input_payload(
    *,
    repository_id: str,
    tree_id: str,
    objective_id: str,
    objective_revision: str,
    policy_id: str,
    policy_revision: str,
    caller: str,
    stage: str,
    goal: Any,
    authority: Any,
    scope: Any,
    acceptance: Any,
    evidence: Iterable[ContextReference] = (),
) -> dict[str, Any]:
    """Return the canonical authority-bearing provider input.

    This is deliberately separate from :class:`ContextCapsule`'s supervisor
    accounting envelope.  It includes every binding and selected reference
    sent to the provider while excluding deferred expansion handles and
    supervisor-only omission diagnostics.
    """

    return {
        "contract_version": CONTEXT_COMPILER_VERSION,
        "repository_id": repository_id,
        "tree_id": tree_id,
        "objective_id": objective_id,
        "objective_revision": objective_revision,
        "policy_id": policy_id,
        "policy_revision": policy_revision,
        "caller": caller,
        "stage": stage,
        **_core_payload(
            goal=goal,
            authority=authority,
            scope=scope,
            acceptance=acceptance,
        ),
        "evidence": tuple(item.to_record() for item in evidence),
    }


@dataclass(frozen=True)
class RequiredContextBudgetEvidence(CanonicalContract):
    """Qualifying witness that required context survived the effective budget."""

    SCHEMA: ClassVar[str] = REQUIRED_CONTEXT_BUDGET_EVIDENCE_SCHEMA

    repository_id: str
    tree_id: str
    policy_id: str
    policy_revision: str
    capsule_id: str
    effective_input_limit: int
    input_tokens: int
    required_fields: tuple[str, ...]
    required_reference_ids: tuple[str, ...]
    selected_reference_ids: tuple[str, ...]
    artifact_digest: str
    requirement_id: str = REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID
    result: str = "passed"

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "policy_id",
            "policy_revision",
            "capsule_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.requirement_id != REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID:
            raise ContextCompilationError("unexpected required-context requirement ID")
        if self.result != "passed":
            raise ContextCompilationError("required-context evidence must pass")
        object.__setattr__(
            self,
            "effective_input_limit",
            _integer(
                self.effective_input_limit,
                "effective_input_limit",
                minimum=1,
            ),
        )
        object.__setattr__(
            self, "input_tokens", _integer(self.input_tokens, "input_tokens")
        )
        if self.input_tokens > self.effective_input_limit:
            raise ContextCompilationError("evidence exceeds effective input limit")
        fields = _strings(self.required_fields, "required_fields")
        if fields != ("acceptance", "authority", "goal", "scope"):
            raise ContextCompilationError(
                "required fields must bind goal, authority, scope, and acceptance"
            )
        object.__setattr__(self, "required_fields", fields)
        required = _strings(
            self.required_reference_ids, "required_reference_ids"
        )
        selected = _strings(
            self.selected_reference_ids, "selected_reference_ids"
        )
        if not set(required).issubset(selected):
            raise ContextCompilationError(
                "required references must be selected by qualifying evidence"
            )
        object.__setattr__(self, "required_reference_ids", required)
        object.__setattr__(self, "selected_reference_ids", selected)
        object.__setattr__(
            self, "artifact_digest", _digest(self.artifact_digest, "artifact_digest")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "requirement_id": self.requirement_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "capsule_id": self.capsule_id,
            "effective_input_limit": self.effective_input_limit,
            "input_tokens": self.input_tokens,
            "required_fields": self.required_fields,
            "required_reference_ids": self.required_reference_ids,
            "selected_reference_ids": self.selected_reference_ids,
            "artifact_digest": self.artifact_digest,
            "result": self.result,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "RequiredContextBudgetEvidence":
        _schema(payload, cls.SCHEMA, "required-context evidence")
        _reject_unknown(
            payload,
            {
                "schema",
                "content_id",
                "contract_version",
                "requirement_id",
                "repository_id",
                "tree_id",
                "policy_id",
                "policy_revision",
                "capsule_id",
                "effective_input_limit",
                "input_tokens",
                "required_fields",
                "required_reference_ids",
                "selected_reference_ids",
                "artifact_digest",
                "result",
            },
            "required-context evidence",
        )
        result = cls(
            requirement_id=payload.get("requirement_id", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            capsule_id=payload.get("capsule_id", ""),
            effective_input_limit=payload.get("effective_input_limit", 0),
            input_tokens=payload.get("input_tokens", 0),
            required_fields=tuple(payload.get("required_fields", ())),
            required_reference_ids=tuple(
                payload.get("required_reference_ids", ())
            ),
            selected_reference_ids=tuple(
                payload.get("selected_reference_ids", ())
            ),
            artifact_digest=payload.get("artifact_digest", ""),
            result=payload.get("result", ""),
        )
        _check_identity(payload, result.content_id, "required-context evidence")
        return result


@dataclass(frozen=True)
class ContextCompilationReceipt(CanonicalContract):
    """Bounded audit receipt for one base-context compilation."""

    SCHEMA: ClassVar[str] = CONTEXT_COMPILATION_RECEIPT_SCHEMA

    repository_id: str
    tree_id: str
    objective_id: str
    policy_id: str
    policy_revision: str
    stage: str
    capsule_id: str
    effective_input_limit: int
    input_tokens: int
    estimator_name: str
    estimator_error_bps: int
    decisions: tuple[EvidenceSelectionDecision, ...] = ()
    evidence: RequiredContextBudgetEvidence | None = None

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "objective_id",
            "policy_id",
            "policy_revision",
            "stage",
            "capsule_id",
            "estimator_name",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("effective_input_limit", "input_tokens", "estimator_error_bps"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.effective_input_limit < 1:
            raise ContextCompilationError("effective_input_limit must be positive")
        if self.input_tokens > self.effective_input_limit:
            raise ContextCompilationError("receipt input exceeds effective limit")
        if self.estimator_error_bps > MAX_ERROR_BPS:
            raise ContextCompilationError("estimator error exceeds its bound")
        decisions: list[EvidenceSelectionDecision] = []
        for raw in self.decisions:
            decisions.append(
                raw
                if isinstance(raw, EvidenceSelectionDecision)
                else EvidenceSelectionDecision.from_dict(raw)
            )
        decisions.sort(key=lambda item: item.reference_id)
        if len(decisions) > MAX_DECISIONS or len(
            {item.reference_id for item in decisions}
        ) != len(decisions):
            raise ContextCompilationError(
                "selection decisions must be bounded and unique"
            )
        object.__setattr__(self, "decisions", tuple(decisions))
        if self.evidence is not None:
            evidence = (
                self.evidence
                if isinstance(self.evidence, RequiredContextBudgetEvidence)
                else RequiredContextBudgetEvidence.from_dict(self.evidence)
            )
            if (
                evidence.repository_id != self.repository_id
                or evidence.tree_id != self.tree_id
                or evidence.policy_id != self.policy_id
                or evidence.policy_revision != self.policy_revision
                or evidence.capsule_id != self.capsule_id
                or evidence.input_tokens != self.input_tokens
                or evidence.effective_input_limit != self.effective_input_limit
            ):
                raise ContextCompilationError(
                    "required-context evidence is not bound to its receipt"
                )
            object.__setattr__(self, "evidence", evidence)

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def evidence_claim_references(self) -> tuple[str, ...]:
        return (
            (REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID,)
            if self.evidence is not None
            else ()
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "stage": self.stage,
            "capsule_id": self.capsule_id,
            "effective_input_limit": self.effective_input_limit,
            "input_tokens": self.input_tokens,
            "estimator_name": self.estimator_name,
            "estimator_error_bps": self.estimator_error_bps,
            "decisions": tuple(item.to_dict() for item in self.decisions),
            "evidence": self.evidence.to_record() if self.evidence else None,
            "evidence_claim_references": self.evidence_claim_references,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextCompilationReceipt":
        _schema(payload, cls.SCHEMA, "context compilation receipt")
        _reject_unknown(
            payload,
            {
                "schema",
                "content_id",
                "receipt_id",
                "contract_version",
                "repository_id",
                "tree_id",
                "objective_id",
                "policy_id",
                "policy_revision",
                "stage",
                "capsule_id",
                "effective_input_limit",
                "input_tokens",
                "estimator_name",
                "estimator_error_bps",
                "decisions",
                "evidence",
                "evidence_claim_references",
            },
            "context compilation receipt",
        )
        evidence = payload.get("evidence")
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            stage=payload.get("stage", ""),
            capsule_id=payload.get("capsule_id", ""),
            effective_input_limit=payload.get("effective_input_limit", 0),
            input_tokens=payload.get("input_tokens", 0),
            estimator_name=payload.get("estimator_name", ""),
            estimator_error_bps=payload.get("estimator_error_bps", 0),
            decisions=tuple(
                EvidenceSelectionDecision.from_dict(item)
                for item in payload.get("decisions", ())
            ),
            evidence=(
                RequiredContextBudgetEvidence.from_dict(evidence)
                if isinstance(evidence, Mapping)
                else None
            ),
        )
        claims = payload.get("evidence_claim_references")
        if claims is not None and _strings(
            claims, "evidence_claim_references"
        ) != result.evidence_claim_references:
            raise ContextCompilationError("context evidence claim is forged")
        _check_identity(payload, result.content_id, "context compilation receipt")
        return result


@dataclass(frozen=True)
class DeltaRetryContextEvidence(CanonicalContract):
    """Qualifying witness for a smaller lossless parent-bound retry delta."""

    SCHEMA: ClassVar[str] = DELTA_RETRY_CONTEXT_EVIDENCE_SCHEMA

    repository_id: str
    tree_id: str
    policy_id: str
    policy_revision: str
    parent_capsule_id: str
    delta_capsule_id: str
    reconstructed_capsule_id: str
    full_replay_tokens: int
    delta_tokens: int
    required_coverage_ids: tuple[str, ...]
    reconstructed_coverage_ids: tuple[str, ...]
    changed_reference_ids: tuple[str, ...]
    requested_reference_ids: tuple[str, ...]
    retained_reference_ids: tuple[str, ...]
    required_fields: tuple[str, ...]
    artifact_digest: str
    requirement_id: str = DELTA_RETRY_EVIDENCE_ID
    result: str = "passed"

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "policy_id",
            "policy_revision",
            "parent_capsule_id",
            "delta_capsule_id",
            "reconstructed_capsule_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.requirement_id != DELTA_RETRY_EVIDENCE_ID:
            raise ContextDeltaError("unexpected delta-retry requirement ID")
        if self.result != "passed":
            raise ContextDeltaError("delta-retry evidence must pass")
        for name in ("full_replay_tokens", "delta_tokens"):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )
        if self.delta_tokens >= self.full_replay_tokens:
            raise ContextDeltaError("qualifying delta must use fewer tokens")
        required = _strings(
            self.required_coverage_ids, "required_coverage_ids"
        )
        reconstructed = _strings(
            self.reconstructed_coverage_ids, "reconstructed_coverage_ids"
        )
        if not set(required).issubset(reconstructed):
            raise ContextDeltaError("qualifying delta loses required coverage")
        object.__setattr__(self, "required_coverage_ids", required)
        object.__setattr__(self, "reconstructed_coverage_ids", reconstructed)
        changed = _strings(
            self.changed_reference_ids, "changed_reference_ids"
        )
        object.__setattr__(self, "changed_reference_ids", changed)
        requested = _strings(
            self.requested_reference_ids, "requested_reference_ids"
        )
        if set(changed).intersection(requested):
            raise ContextDeltaError(
                "changed and requested-only references must be disjoint"
            )
        if not changed and not requested:
            raise ContextDeltaError(
                "qualifying delta must carry changed or requested evidence"
            )
        object.__setattr__(self, "requested_reference_ids", requested)
        retained = _strings(
            self.retained_reference_ids, "retained_reference_ids"
        )
        if set(changed).union(requested).intersection(retained):
            raise ContextDeltaError(
                "transmitted and retained delta references must be disjoint"
            )
        object.__setattr__(self, "retained_reference_ids", retained)
        fields = _strings(self.required_fields, "required_fields")
        if fields != ("acceptance", "authority", "goal", "scope"):
            raise ContextDeltaError(
                "delta evidence must preserve every invariant context field"
            )
        object.__setattr__(self, "required_fields", fields)
        object.__setattr__(
            self, "artifact_digest", _digest(self.artifact_digest, "artifact_digest")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "requirement_id": self.requirement_id,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "parent_capsule_id": self.parent_capsule_id,
            "delta_capsule_id": self.delta_capsule_id,
            "reconstructed_capsule_id": self.reconstructed_capsule_id,
            "full_replay_tokens": self.full_replay_tokens,
            "delta_tokens": self.delta_tokens,
            "required_coverage_ids": self.required_coverage_ids,
            "reconstructed_coverage_ids": self.reconstructed_coverage_ids,
            "changed_reference_ids": self.changed_reference_ids,
            "requested_reference_ids": self.requested_reference_ids,
            "retained_reference_ids": self.retained_reference_ids,
            "required_fields": self.required_fields,
            "artifact_digest": self.artifact_digest,
            "result": self.result,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "DeltaRetryContextEvidence":
        _schema(payload, cls.SCHEMA, "delta-retry evidence")
        _reject_unknown(
            payload,
            {
                "schema",
                "content_id",
                "contract_version",
                "requirement_id",
                "repository_id",
                "tree_id",
                "policy_id",
                "policy_revision",
                "parent_capsule_id",
                "delta_capsule_id",
                "reconstructed_capsule_id",
                "full_replay_tokens",
                "delta_tokens",
                "required_coverage_ids",
                "reconstructed_coverage_ids",
                "changed_reference_ids",
                "requested_reference_ids",
                "retained_reference_ids",
                "required_fields",
                "artifact_digest",
                "result",
            },
            "delta-retry evidence",
        )
        result = cls(
            requirement_id=payload.get("requirement_id", ""),
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            parent_capsule_id=payload.get("parent_capsule_id", ""),
            delta_capsule_id=payload.get("delta_capsule_id", ""),
            reconstructed_capsule_id=payload.get(
                "reconstructed_capsule_id", ""
            ),
            full_replay_tokens=payload.get("full_replay_tokens", 0),
            delta_tokens=payload.get("delta_tokens", 0),
            required_coverage_ids=tuple(
                payload.get("required_coverage_ids", ())
            ),
            reconstructed_coverage_ids=tuple(
                payload.get("reconstructed_coverage_ids", ())
            ),
            changed_reference_ids=tuple(
                payload.get("changed_reference_ids", ())
            ),
            requested_reference_ids=tuple(
                payload.get("requested_reference_ids", ())
            ),
            retained_reference_ids=tuple(
                payload.get("retained_reference_ids", ())
            ),
            required_fields=tuple(payload.get("required_fields", ())),
            artifact_digest=payload.get("artifact_digest", ""),
            result=payload.get("result", ""),
        )
        _check_identity(payload, result.content_id, "delta-retry evidence")
        return result


@dataclass(frozen=True)
class ContextDeltaReceipt(CanonicalContract):
    """Content-addressed audit receipt for one retry delta."""

    SCHEMA: ClassVar[str] = CONTEXT_DELTA_RECEIPT_SCHEMA

    repository_id: str
    tree_id: str
    objective_id: str
    policy_id: str
    policy_revision: str
    parent_capsule_id: str
    delta_capsule_id: str
    reconstructed_capsule_id: str
    full_replay_tokens: int
    delta_tokens: int
    decisions: tuple[EvidenceSelectionDecision, ...]
    evidence: DeltaRetryContextEvidence | None = None

    def __post_init__(self) -> None:
        for name in (
            "repository_id",
            "tree_id",
            "objective_id",
            "policy_id",
            "policy_revision",
            "parent_capsule_id",
            "delta_capsule_id",
            "reconstructed_capsule_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in ("full_replay_tokens", "delta_tokens"):
            object.__setattr__(
                self, name, _integer(getattr(self, name), name, minimum=1)
            )
        decisions = tuple(
            sorted(
                (
                    item
                    if isinstance(item, EvidenceSelectionDecision)
                    else EvidenceSelectionDecision.from_dict(item)
                    for item in self.decisions
                ),
                key=lambda item: item.reference_id,
            )
        )
        if len(decisions) > MAX_DECISIONS or len(
            {item.reference_id for item in decisions}
        ) != len(decisions):
            raise ContextDeltaError("delta decisions must be bounded and unique")
        object.__setattr__(self, "decisions", decisions)
        if self.evidence is not None:
            evidence = (
                self.evidence
                if isinstance(self.evidence, DeltaRetryContextEvidence)
                else DeltaRetryContextEvidence.from_dict(self.evidence)
            )
            if any(
                (
                    evidence.repository_id != self.repository_id,
                    evidence.tree_id != self.tree_id,
                    evidence.policy_id != self.policy_id,
                    evidence.policy_revision != self.policy_revision,
                    evidence.parent_capsule_id != self.parent_capsule_id,
                    evidence.delta_capsule_id != self.delta_capsule_id,
                    evidence.reconstructed_capsule_id
                    != self.reconstructed_capsule_id,
                    evidence.full_replay_tokens != self.full_replay_tokens,
                    evidence.delta_tokens != self.delta_tokens,
                )
            ):
                raise ContextDeltaError("delta evidence is not bound to its receipt")
            included = {
                item.reference_id
                for item in decisions
                if item.included
            }
            witnessed = set(evidence.changed_reference_ids).union(
                evidence.requested_reference_ids
            )
            if included != witnessed:
                raise ContextDeltaError(
                    "delta decisions do not match witnessed transmitted references"
                )
            object.__setattr__(self, "evidence", evidence)

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def evidence_claim_references(self) -> tuple[str, ...]:
        return (DELTA_RETRY_EVIDENCE_ID,) if self.evidence else ()

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTEXT_COMPILER_VERSION,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "objective_id": self.objective_id,
            "policy_id": self.policy_id,
            "policy_revision": self.policy_revision,
            "parent_capsule_id": self.parent_capsule_id,
            "delta_capsule_id": self.delta_capsule_id,
            "reconstructed_capsule_id": self.reconstructed_capsule_id,
            "full_replay_tokens": self.full_replay_tokens,
            "delta_tokens": self.delta_tokens,
            "decisions": tuple(item.to_dict() for item in self.decisions),
            "evidence": self.evidence.to_record() if self.evidence else None,
            "evidence_claim_references": self.evidence_claim_references,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextDeltaReceipt":
        _schema(payload, cls.SCHEMA, "context delta receipt")
        _reject_unknown(
            payload,
            {
                "schema",
                "content_id",
                "receipt_id",
                "contract_version",
                "repository_id",
                "tree_id",
                "objective_id",
                "policy_id",
                "policy_revision",
                "parent_capsule_id",
                "delta_capsule_id",
                "reconstructed_capsule_id",
                "full_replay_tokens",
                "delta_tokens",
                "decisions",
                "evidence",
                "evidence_claim_references",
            },
            "context delta receipt",
        )
        evidence = payload.get("evidence")
        result = cls(
            repository_id=payload.get("repository_id", ""),
            tree_id=payload.get("tree_id", ""),
            objective_id=payload.get("objective_id", ""),
            policy_id=payload.get("policy_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            parent_capsule_id=payload.get("parent_capsule_id", ""),
            delta_capsule_id=payload.get("delta_capsule_id", ""),
            reconstructed_capsule_id=payload.get(
                "reconstructed_capsule_id", ""
            ),
            full_replay_tokens=payload.get("full_replay_tokens", 0),
            delta_tokens=payload.get("delta_tokens", 0),
            decisions=tuple(
                EvidenceSelectionDecision.from_dict(item)
                for item in payload.get("decisions", ())
            ),
            evidence=(
                DeltaRetryContextEvidence.from_dict(evidence)
                if isinstance(evidence, Mapping)
                else None
            ),
        )
        claims = payload.get("evidence_claim_references")
        if claims is not None and _strings(
            claims, "evidence_claim_references"
        ) != result.evidence_claim_references:
            raise ContextDeltaError("delta evidence claim is forged")
        _check_identity(payload, result.content_id, "context delta receipt")
        return result


@dataclass(frozen=True)
class ContextCompileResult:
    capsule: ContextCapsule
    receipt: ContextCompilationReceipt
    decisions: tuple[EvidenceSelectionDecision, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.capsule, ContextCapsule):
            raise ContextCompilationError("capsule must be a ContextCapsule")
        if not isinstance(self.receipt, ContextCompilationReceipt):
            raise ContextCompilationError(
                "receipt must be a ContextCompilationReceipt"
            )
        receipt_bindings = {
            "repository_id": self.capsule.repository_id,
            "tree_id": self.capsule.tree_id,
            "objective_id": self.capsule.objective_id,
            "policy_id": self.capsule.policy_id,
            "policy_revision": self.capsule.policy_revision,
            "stage": self.capsule.stage,
            "capsule_id": self.capsule.capsule_id,
            "effective_input_limit": self.capsule.budget.max_input_tokens,
            "input_tokens": self.capsule.input_tokens,
        }
        if any(
            getattr(self.receipt, name) != expected
            for name, expected in receipt_bindings.items()
        ):
            raise ContextCompilationError(
                "receipt does not bind the complete compiled capsule"
            )
        if self.decisions != self.receipt.decisions:
            raise ContextCompilationError("result decisions do not match receipt")
        evidence = self.receipt.evidence
        if evidence is None:
            raise ContextCompilationError(
                "compiled result requires qualifying required-context evidence"
            )
        selected_by_id = {
            item.reference_id: item for item in self.capsule.evidence
        }
        expansion_by_id = {
            item.reference_id: item
            for item in self.capsule.expansion_references
        }
        decision_by_id = {
            item.reference_id: item for item in self.decisions
        }
        if set(decision_by_id) != set(selected_by_id) | set(expansion_by_id):
            raise ContextCompilationError(
                "selection decisions do not cover the complete candidate set"
            )
        required_ids = {
            item.reference_id
            for item in self.capsule.evidence
            if item.required
        }
        if set(evidence.required_reference_ids) != required_ids:
            raise ContextCompilationError(
                "required-context evidence does not bind required references"
            )
        if set(evidence.selected_reference_ids) != set(selected_by_id):
            raise ContextCompilationError(
                "required-context evidence does not bind selected references"
            )
        if evidence.required_fields != tuple(
            sorted(self.capsule.required_field_names)
        ):
            raise ContextCompilationError(
                "required-context evidence does not bind invariant fields"
            )
        for reference_id, reference in selected_by_id.items():
            decision = decision_by_id[reference_id]
            expected_reason = (
                InclusionReason.REQUIRED
                if reference.required
                else InclusionReason.RANKED_FIT
            )
            if (
                not decision.included
                or decision.reason is not expected_reason
                or decision.priority != reference.priority
            ):
                raise ContextCompilationError(
                    "selection decision does not match selected reference"
                )
        for reference_id, reference in expansion_by_id.items():
            decision = decision_by_id[reference_id]
            if (
                decision.included
                or decision.reason
                not in (ExclusionReason.TOKEN_BUDGET, ExclusionReason.ITEM_LIMIT)
                or decision.priority != reference.priority
            ):
                raise ContextCompilationError(
                    "selection decision does not match deferred reference"
                )
        if evidence.artifact_digest != _canonical_digest(
            self.capsule.to_record()
        ):
            raise ContextCompilationError(
                "required-context evidence artifact digest does not match capsule"
            )


@dataclass(frozen=True)
class ContextDeltaResult:
    delta_capsule: ContextDeltaCapsule
    reconstructed_capsule: ContextCapsule
    receipt: ContextDeltaReceipt
    decisions: tuple[EvidenceSelectionDecision, ...]

    @property
    def capsule(self) -> ContextDeltaCapsule:
        return self.delta_capsule

    def __post_init__(self) -> None:
        if (
            self.receipt.delta_capsule_id != self.delta_capsule.capsule_id
            or self.receipt.reconstructed_capsule_id
            != self.reconstructed_capsule.capsule_id
            or self.decisions != self.receipt.decisions
        ):
            raise ContextDeltaError("delta result is not receipt-bound")
        evidence = self.receipt.evidence
        if evidence is None:
            raise ContextDeltaError(
                "delta result requires qualifying delta-retry evidence"
            )
        transmitted_ids = {
            item.reference_id for item in self.delta_capsule.evidence
        }
        witnessed_ids = set(evidence.changed_reference_ids).union(
            evidence.requested_reference_ids
        )
        if transmitted_ids != witnessed_ids:
            raise ContextDeltaError(
                "delta witness does not describe the transmitted references"
            )
        if (
            self.delta_capsule.requested_reference_ids
            != evidence.requested_reference_ids
        ):
            raise ContextDeltaError(
                "delta witness does not bind explicitly requested references"
            )
        retained_ids = {
            item.reference_id for item in self.reconstructed_capsule.evidence
        }.difference(transmitted_ids)
        if set(evidence.retained_reference_ids) != retained_ids:
            raise ContextDeltaError(
                "delta witness does not bind retained references"
            )
        required_coverage = {
            coverage
            for item in self.reconstructed_capsule.evidence
            if item.required
            for coverage in item.coverage_ids
        }
        if set(evidence.required_coverage_ids) != required_coverage:
            raise ContextDeltaError(
                "delta witness does not bind reconstructed required coverage"
            )
        if (
            evidence.reconstructed_coverage_ids
            != self.reconstructed_capsule.evidence_coverage_ids
            or evidence.required_fields
            != tuple(sorted(self.reconstructed_capsule.required_field_names))
        ):
            raise ContextDeltaError(
                "delta witness does not bind reconstructed context coverage"
            )
        expected_digest = _canonical_digest(
            {
                "parent_capsule_id": self.delta_capsule.parent_capsule_id,
                "delta": self.delta_capsule.to_record(),
                "reconstructed": self.reconstructed_capsule.to_record(),
            }
        )
        if evidence.artifact_digest != expected_digest:
            raise ContextDeltaError(
                "delta witness artifact digest does not match its capsules"
            )


class ContextCompiler:
    """Compile base and retry contexts under one provider-aware budget."""

    def __init__(
        self,
        budget: ContextBudget,
        *,
        tokenizer: Callable[[str], Any] | Any | None = None,
        estimator: CalibratedTokenEstimator | None = None,
        provider_context_window: int | None = None,
        provider_max_input_tokens: int | None = None,
        reserved_output_tokens: int | None = None,
        reserved_tool_tokens: int | None = None,
    ) -> None:
        if not isinstance(budget, ContextBudget):
            if not isinstance(budget, Mapping):
                raise ContextCompilationError("budget must be a ContextBudget")
            budget = ContextBudget.from_dict(budget)
        if estimator is not None and tokenizer is not None:
            raise ContextCompilationError(
                "provide tokenizer or estimator, not both"
            )
        self.budget = budget
        self.estimator = estimator or CalibratedTokenEstimator(tokenizer)
        self.effective_input_limit = budget.effective_input_limit(
            provider_context_window=provider_context_window,
            provider_max_input_tokens=provider_max_input_tokens,
            reserved_output_tokens=reserved_output_tokens,
            reserved_tool_tokens=reserved_tool_tokens,
        )
        if self.effective_input_limit < 1:
            raise RequiredContextOverflowError(
                "provider reserves leave no usable input budget"
            )
        self.effective_budget = budget.for_effective_input_limit(
            self.effective_input_limit
        )

    def _provider_input_tokens(
        self,
        *,
        repository_id: str,
        tree_id: str,
        objective_id: str,
        objective_revision: str,
        policy_id: str,
        policy_revision: str,
        caller: str,
        stage: str,
        goal: Any,
        authority: Any,
        scope: Any,
        acceptance: Any,
        evidence: Iterable[ContextReference] = (),
    ) -> int:
        selected = tuple(evidence)
        canonical_count = self.estimator.estimate(
            context_provider_input_payload(
                repository_id=repository_id,
                tree_id=tree_id,
                objective_id=objective_id,
                objective_revision=objective_revision,
                policy_id=policy_id,
                policy_revision=policy_revision,
                caller=caller,
                stage=stage,
                goal=goal,
                authority=authority,
                scope=scope,
                acceptance=acceptance,
                evidence=selected,
            )
        )
        # Component accounting is conservative for tokenizers whose result is
        # not additive across JSON boundaries and for references carrying a
        # larger producer-observed count than the local tokenizer.
        component_count = self.estimator.estimate(
            context_provider_input_payload(
                repository_id=repository_id,
                tree_id=tree_id,
                objective_id=objective_id,
                objective_revision=objective_revision,
                policy_id=policy_id,
                policy_revision=policy_revision,
                caller=caller,
                stage=stage,
                goal=goal,
                authority=authority,
                scope=scope,
                acceptance=acceptance,
                evidence=(),
            )
        ) + sum(_reference_tokens(self.estimator, item) for item in selected)
        return max(canonical_count, component_count)

    def estimate_capsule_input(self, capsule: ContextCapsule) -> int:
        """Independently recompute conservative provider input accounting."""

        if not isinstance(capsule, ContextCapsule):
            raise ContextCompilationError("capsule must be a ContextCapsule")
        return self._provider_input_tokens(
            repository_id=capsule.repository_id,
            tree_id=capsule.tree_id,
            objective_id=capsule.objective_id,
            objective_revision=capsule.objective_revision,
            policy_id=capsule.policy_id,
            policy_revision=capsule.policy_revision,
            caller=capsule.caller,
            stage=capsule.stage,
            goal=capsule.goal,
            authority=capsule.authority,
            scope=capsule.scope,
            acceptance=capsule.acceptance,
            evidence=capsule.evidence,
        )

    def compile(
        self,
        *,
        repository_id: str,
        tree_id: str,
        objective_id: str,
        objective_revision: str,
        policy_id: str,
        policy_revision: str,
        caller: str,
        stage: str,
        goal: Any,
        authority: Any,
        scope: Any,
        acceptance: Any,
        evidence: Iterable[ContextReference | Mapping[str, Any]] = (),
    ) -> ContextCompileResult:
        references = _coerce_references(evidence)
        required = tuple(item for item in references if item.required)
        optional = tuple(item for item in references if not item.required)
        input_arguments = {
            "repository_id": repository_id,
            "tree_id": tree_id,
            "objective_id": objective_id,
            "objective_revision": objective_revision,
            "policy_id": policy_id,
            "policy_revision": policy_revision,
            "caller": caller,
            "stage": stage,
            "goal": goal,
            "authority": authority,
            "scope": scope,
            "acceptance": acceptance,
        }
        base_tokens = self._provider_input_tokens(
            **input_arguments,
            evidence=(),
        )
        selected: list[ContextReference] = []
        decisions: dict[str, EvidenceSelectionDecision] = {}
        used = base_tokens
        if used > self.effective_input_limit:
            raise RequiredContextOverflowError(
                "invariant goal/authority/scope/acceptance exceeds "
                "the effective provider input budget"
            )
        for item in sorted(required, key=lambda member: member.reference_id):
            tokens = _reference_tokens(self.estimator, item)
            proposed = self._provider_input_tokens(
                **input_arguments,
                evidence=(*selected, item),
            )
            if (
                len(selected) >= self.effective_budget.max_items
                or proposed > self.effective_input_limit
            ):
                raise RequiredContextOverflowError(
                    f"required evidence {item.reference_id!r} does not fit "
                    "the effective provider input budget"
                )
            selected.append(item)
            used = proposed
            decisions[item.reference_id] = EvidenceSelectionDecision(
                item.reference_id,
                True,
                InclusionReason.REQUIRED,
                tokens,
                item.priority,
            )
        ranked_optional = sorted(
            optional,
            key=lambda item: (
                -item.priority,
                item.tier.value,
                item.reference_id,
                item.reference_content_id,
            ),
        )
        omitted: list[ContextReference] = []
        for item in ranked_optional:
            tokens = _reference_tokens(self.estimator, item)
            proposed = self._provider_input_tokens(
                **input_arguments,
                evidence=(*selected, item),
            )
            if len(selected) >= self.effective_budget.max_items:
                reason = ExclusionReason.ITEM_LIMIT
            elif proposed > self.effective_input_limit:
                reason = ExclusionReason.TOKEN_BUDGET
            else:
                selected.append(item)
                used = proposed
                decisions[item.reference_id] = EvidenceSelectionDecision(
                    item.reference_id,
                    True,
                    InclusionReason.RANKED_FIT,
                    tokens,
                    item.priority,
                )
                continue
            omitted.append(item)
            decisions[item.reference_id] = EvidenceSelectionDecision(
                item.reference_id,
                False,
                reason,
                tokens,
                item.priority,
            )
        ordered_decisions = tuple(
            decisions[key] for key in sorted(decisions)
        )
        capsule = ContextCapsule(
            repository_id=repository_id,
            tree_id=tree_id,
            objective_id=objective_id,
            objective_revision=objective_revision,
            policy_id=policy_id,
            policy_revision=policy_revision,
            caller=caller,
            stage=stage,
            budget=self.effective_budget,
            goal=goal,
            authority=authority,
            scope=scope,
            acceptance=acceptance,
            evidence=tuple(selected),
            expansion_references=tuple(_as_expansion(item) for item in omitted),
            input_tokens=used,
            truncated=bool(omitted),
            omissions=tuple(
                f"{item.reference_id}:{decisions[item.reference_id].reason.value}"
                for item in omitted
            ),
        )
        selected_ids = tuple(item.reference_id for item in capsule.evidence)
        required_ids = tuple(item.reference_id for item in required)
        witness = RequiredContextBudgetEvidence(
            repository_id=capsule.repository_id,
            tree_id=capsule.tree_id,
            policy_id=capsule.policy_id,
            policy_revision=capsule.policy_revision,
            capsule_id=capsule.capsule_id,
            effective_input_limit=self.effective_input_limit,
            input_tokens=capsule.input_tokens,
            required_fields=capsule.required_field_names,
            required_reference_ids=required_ids,
            selected_reference_ids=selected_ids,
            artifact_digest=_canonical_digest(capsule.to_record()),
        )
        receipt = ContextCompilationReceipt(
            repository_id=capsule.repository_id,
            tree_id=capsule.tree_id,
            objective_id=capsule.objective_id,
            policy_id=capsule.policy_id,
            policy_revision=capsule.policy_revision,
            stage=capsule.stage,
            capsule_id=capsule.capsule_id,
            effective_input_limit=self.effective_input_limit,
            input_tokens=capsule.input_tokens,
            estimator_name=self.estimator.name,
            estimator_error_bps=self.estimator.error_bps,
            decisions=ordered_decisions,
            evidence=witness,
        )
        return ContextCompileResult(capsule, receipt, ordered_decisions)

    compile_context = compile

    def compile_delta(
        self,
        parent: ContextCapsule,
        *,
        evidence: Iterable[ContextReference | Mapping[str, Any]],
        requested_reference_ids: Iterable[str] = (),
        stage: str | None = None,
    ) -> ContextDeltaResult:
        if not isinstance(parent, ContextCapsule):
            raise ContextDeltaError("parent must be a ContextCapsule")
        if parent.is_delta:
            raise ContextDeltaError(
                "delta chaining requires reconstruction of the prior delta"
            )
        candidates = _coerce_references(evidence)
        candidate_by_id = {item.reference_id: item for item in candidates}
        parent_by_id = {item.reference_id: item for item in parent.evidence}
        requested = set(
            _strings(requested_reference_ids, "requested_reference_ids")
        )
        unknown_requests = requested.difference(candidate_by_id)
        if unknown_requests:
            raise ContextDeltaError(
                "requested retry evidence is not present in the candidate set"
            )
        required_ids = {
            item.reference_id
            for item in parent.evidence
            if item.required
        }
        missing_required = required_ids.difference(candidate_by_id)
        if missing_required:
            raise ContextDeltaError(
                "retry candidate drops required evidence references"
            )
        downgraded_required = {
            reference_id
            for reference_id in required_ids
            if not candidate_by_id[reference_id].required
        }
        if downgraded_required:
            raise ContextDeltaError(
                "retry candidate downgrades parent-required evidence"
            )
        transmitted: list[ContextReference] = []
        genuinely_changed: list[str] = []
        requested_only: list[str] = []
        decisions: list[EvidenceSelectionDecision] = []
        for item in candidates:
            previous = parent_by_id.get(item.reference_id)
            is_changed = (
                previous is None
                or previous.reference_content_id != item.reference_content_id
                or previous.to_dict() != item.to_dict()
            )
            tokens = _reference_tokens(self.estimator, item)
            if is_changed or item.reference_id in requested:
                transmitted.append(item)
                if is_changed:
                    genuinely_changed.append(item.reference_id)
                else:
                    requested_only.append(item.reference_id)
                decisions.append(
                    EvidenceSelectionDecision(
                        item.reference_id,
                        True,
                        (
                            InclusionReason.CHANGED
                            if is_changed
                            else InclusionReason.REQUESTED
                        ),
                        tokens,
                        item.priority,
                    )
                )
            else:
                decisions.append(
                    EvidenceSelectionDecision(
                        item.reference_id,
                        False,
                        ExclusionReason.UNCHANGED,
                        tokens,
                        item.priority,
                    )
                )
        if not transmitted:
            raise ContextDeltaError(
                "retry delta must contain changed or explicitly requested evidence"
            )
        combined_by_id = dict(parent_by_id)
        combined_by_id.update(candidate_by_id)
        combined = tuple(combined_by_id[key] for key in sorted(combined_by_id))
        reconstructed_input_tokens = self._provider_input_tokens(
            repository_id=parent.repository_id,
            tree_id=parent.tree_id,
            objective_id=parent.objective_id,
            objective_revision=parent.objective_revision,
            policy_id=parent.policy_id,
            policy_revision=parent.policy_revision,
            caller=parent.caller,
            stage=stage or parent.stage,
            goal=parent.goal,
            authority=parent.authority,
            scope=parent.scope,
            acceptance=parent.acceptance,
            evidence=combined,
        )
        reconstructed_limit = min(
            parent.budget.max_input_tokens, self.effective_input_limit
        )
        if reconstructed_input_tokens > reconstructed_limit:
            raise ContextDeltaError(
                "reconstructed full context exceeds the effective input budget"
            )
        delta_capsule = ContextDeltaCapsule(
            parent_capsule_id=parent.capsule_id,
            stage=stage or parent.stage,
            evidence=tuple(transmitted),
            reconstructed_input_tokens=reconstructed_input_tokens,
            requested_reference_ids=tuple(requested_only),
        )
        if len(delta_capsule.canonical_bytes()) > (
            self.effective_budget.max_serialized_bytes
        ):
            raise ContextDeltaError(
                "retry delta exceeds the serialized-byte budget"
            )
        reconstructed = reconstruct_context(parent, delta_capsule)
        # The delta is its compact provider wire record.  Full replay is the
        # canonical provider input, conservatively floored by the same
        # component accounting used for the effective-budget check; it does
        # not include supervisor-only budget or omission metadata.
        delta_tokens = self.estimator.estimate(delta_capsule.to_record())
        full_replay_tokens = max(
            reconstructed.input_tokens,
            self.estimator.estimate(reconstructed.provider_input_payload),
        )
        if delta_tokens >= full_replay_tokens:
            raise ContextDeltaError(
                "retry delta does not use fewer tokens than full replay"
            )
        if delta_tokens > self.effective_input_limit:
            raise ContextDeltaError("retry delta exceeds effective input budget")
        parent_required_coverage = {
            coverage
            for item in parent.evidence
            if item.required
            for coverage in item.coverage_ids
        }
        required_coverage = {
            coverage
            for item in reconstructed.evidence
            if item.required
            for coverage in item.coverage_ids
        }
        reconstructed_coverage = set(reconstructed.evidence_coverage_ids)
        if (
            not parent_required_coverage.issubset(required_coverage)
            or not required_coverage.issubset(reconstructed_coverage)
        ):
            raise ContextDeltaError("retry reconstruction loses required coverage")
        ordered_decisions = tuple(
            sorted(decisions, key=lambda item: item.reference_id)
        )
        witness = DeltaRetryContextEvidence(
            repository_id=parent.repository_id,
            tree_id=parent.tree_id,
            policy_id=parent.policy_id,
            policy_revision=parent.policy_revision,
            parent_capsule_id=parent.capsule_id,
            delta_capsule_id=delta_capsule.capsule_id,
            reconstructed_capsule_id=reconstructed.capsule_id,
            full_replay_tokens=full_replay_tokens,
            delta_tokens=delta_tokens,
            required_coverage_ids=tuple(required_coverage),
            reconstructed_coverage_ids=reconstructed.evidence_coverage_ids,
            changed_reference_ids=tuple(genuinely_changed),
            requested_reference_ids=tuple(requested_only),
            retained_reference_ids=tuple(
                item.reference_id
                for item in reconstructed.evidence
                if item.reference_id
                not in {member.reference_id for member in transmitted}
            ),
            required_fields=reconstructed.required_field_names,
            artifact_digest=_canonical_digest(
                {
                    "parent_capsule_id": parent.capsule_id,
                    "delta": delta_capsule.to_record(),
                    "reconstructed": reconstructed.to_record(),
                }
            ),
        )
        receipt = ContextDeltaReceipt(
            repository_id=parent.repository_id,
            tree_id=parent.tree_id,
            objective_id=parent.objective_id,
            policy_id=parent.policy_id,
            policy_revision=parent.policy_revision,
            parent_capsule_id=parent.capsule_id,
            delta_capsule_id=delta_capsule.capsule_id,
            reconstructed_capsule_id=reconstructed.capsule_id,
            full_replay_tokens=full_replay_tokens,
            delta_tokens=delta_tokens,
            decisions=ordered_decisions,
            evidence=witness,
        )
        return ContextDeltaResult(
            delta_capsule,
            reconstructed,
            receipt,
            ordered_decisions,
        )

    compile_retry = compile_delta


def reconstruct_context(
    parent: ContextCapsule, delta: ContextDeltaCapsule
) -> ContextCapsule:
    """Apply a parent-bound delta and return the deterministic full context."""

    if not isinstance(parent, ContextCapsule) or not isinstance(
        delta, ContextDeltaCapsule
    ):
        raise ContextDeltaError(
            "parent must be a ContextCapsule and delta a ContextDeltaCapsule"
        )
    if delta.parent_capsule_id != parent.capsule_id:
        raise ContextDeltaError("delta is not bound to the supplied parent")
    for item in delta.evidence:
        if item.repository_id and item.repository_id != parent.repository_id:
            raise ContextDeltaError(
                "delta evidence changes immutable repository identity"
            )
        if item.tree_id and item.tree_id != parent.tree_id:
            raise ContextDeltaError(
                "delta evidence changes immutable tree identity"
            )
    combined = {item.reference_id: item for item in parent.evidence}
    combined.update({item.reference_id: item for item in delta.evidence})
    required_ids = {
        item.reference_id for item in parent.evidence if item.required
    }
    if not required_ids.issubset(combined) or any(
        not combined[reference_id].required for reference_id in required_ids
    ):
        raise ContextDeltaError(
            "reconstructed context loses or downgrades required evidence"
        )
    evidence = tuple(combined[key] for key in sorted(combined))
    reconstructed_tokens = delta.reconstructed_input_tokens
    if reconstructed_tokens > parent.budget.max_input_tokens:
        raise ContextDeltaError(
            "reconstructed context exceeds the parent input budget"
        )
    parent_declared_reference_tokens = sum(
        item.token_count for item in parent.evidence
    )
    inherited_core_floor = max(
        0, parent.input_tokens - parent_declared_reference_tokens
    )
    reconstructed_floor = inherited_core_floor + sum(
        item.token_count for item in evidence
    )
    if reconstructed_floor > reconstructed_tokens:
        raise ContextDeltaError(
            "reconstructed token count omits inherited core or evidence tokens"
        )
    selected_ids = set(combined)
    expansions = {
        item.reference_id: item
        for item in parent.expansion_references
        if item.reference_id not in selected_ids
    }
    return ContextCapsule(
        repository_id=parent.repository_id,
        tree_id=parent.tree_id,
        objective_id=parent.objective_id,
        objective_revision=parent.objective_revision,
        policy_id=parent.policy_id,
        policy_revision=parent.policy_revision,
        caller=parent.caller,
        stage=delta.stage,
        budget=parent.budget,
        goal=parent.goal,
        authority=parent.authority,
        scope=parent.scope,
        acceptance=parent.acceptance,
        evidence=evidence,
        expansion_references=tuple(
            expansions[key] for key in sorted(expansions)
        ),
        input_tokens=reconstructed_tokens,
    )


def compile_context_capsule(
    budget: ContextBudget,
    **kwargs: Any,
) -> ContextCompileResult:
    """Convenience wrapper around :class:`ContextCompiler`."""

    compiler_options = {
        key: kwargs.pop(key)
        for key in tuple(kwargs)
        if key
        in {
            "tokenizer",
            "estimator",
            "provider_context_window",
            "provider_max_input_tokens",
            "reserved_output_tokens",
            "reserved_tool_tokens",
        }
    }
    return ContextCompiler(budget, **compiler_options).compile(**kwargs)


def compile_context_delta(
    budget: ContextBudget,
    parent: ContextCapsule,
    **kwargs: Any,
) -> ContextDeltaResult:
    """Convenience wrapper for a parent-bound retry delta."""

    compiler_options = {
        key: kwargs.pop(key)
        for key in tuple(kwargs)
        if key
        in {
            "tokenizer",
            "estimator",
            "provider_context_window",
            "provider_max_input_tokens",
            "reserved_output_tokens",
            "reserved_tool_tokens",
        }
    }
    return ContextCompiler(budget, **compiler_options).compile_delta(
        parent, **kwargs
    )


def expand_context(
    compiler: ContextCompiler,
    parent: ContextCapsule,
    references: Iterable[ContextReference | Mapping[str, Any]],
) -> ContextDeltaResult:
    """Request selected expansion handles as a lossless retry delta."""

    selected = _coerce_references(references)
    return compiler.compile_delta(
        parent,
        evidence=tuple(parent.evidence) + selected,
        requested_reference_ids=tuple(item.reference_id for item in selected),
    )


build_context_capsule = compile_context_capsule
build_context_delta = compile_context_delta
ContextCompilationResult = ContextCompileResult
ContextRetryResult = ContextDeltaResult
reconstruct_context_capsule = reconstruct_context


__all__ = [
    "CONTEXT_COMPILATION_RECEIPT_SCHEMA",
    "CONTEXT_COMPILER_VERSION",
    "CONTEXT_DELTA_RECEIPT_SCHEMA",
    "CONTEXT_EVIDENCE_PRODUCERS",
    "DELTA_RETRY_CONTEXT_EVIDENCE_SCHEMA",
    "DELTA_RETRY_EVIDENCE_ID",
    "REQUIRED_CONTEXT_BUDGET_EVIDENCE_ID",
    "REQUIRED_CONTEXT_BUDGET_EVIDENCE_SCHEMA",
    "CalibratedTokenEstimator",
    "ContextCompilationError",
    "ContextCompilationReceipt",
    "ContextCompilationResult",
    "ContextCompileResult",
    "ContextCompiler",
    "ContextDeltaError",
    "ContextDeltaReceipt",
    "ContextDeltaResult",
    "ContextRetryResult",
    "DeltaRetryContextEvidence",
    "EvidenceSelectionDecision",
    "ExclusionReason",
    "InclusionReason",
    "RequiredContextBudgetEvidence",
    "RequiredContextOverflowError",
    "build_context_capsule",
    "build_context_delta",
    "compile_context_capsule",
    "compile_context_delta",
    "context_provider_input_payload",
    "expand_context",
    "reconstruct_context",
    "reconstruct_context_capsule",
]
