"""Closed, canonical escalation questions for deterministic-first routing.

An :class:`UnresolvedQuestion` describes a question which remains after the
deterministic stages have run.  It is an admission record only: validating a
record neither dispatches a model nor authorizes the resulting decision.

The question identity is a SHA-256 digest of the complete non-identity body.
This makes semantically identical, canonically ordered requests share an
identity while preventing a caller from assigning an unrelated one.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import HarnessError


UNRESOLVED_QUESTION_INTERFACE: Final[str] = "UnresolvedQuestion@1"
UNRESOLVED_QUESTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/unresolved-question@1"
)
UNRESOLVED_QUESTION_SCHEMA_PATH: Final[Path] = (
    Path(__file__).resolve().parent / "schemas" / "unresolved_question.schema.json"
)
UNRESOLVED_QUESTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "question_id",
        "exact_question",
        "why_prior_deterministic_stages_could_not_resolve",
        "evidence_available",
        "evidence_missing",
        "candidate_decisions_answer_could_change",
        "minimum_model_capability",
        "context_budget",
        "response_schema",
        "deadline",
        "cost_budget",
    }
)

MAX_QUESTION_CHARS: Final[int] = 4_096
MAX_REASON_CHARS: Final[int] = 2_048
MAX_EVIDENCE_CHARS: Final[int] = 1_024
MAX_RESPONSE_VALUE_CHARS: Final[int] = 256
MAX_EVIDENCE_ITEMS: Final[int] = 64
MAX_RESPONSE_VALUES: Final[int] = 16
MAX_CONTEXT_BUDGET: Final[int] = 262_144
MAX_COST_BUDGET_MICROUSD: Final[int] = 1_000_000_000_000
MIN_DEADLINE_YEAR: Final[int] = 2000
MAX_DEADLINE_YEAR: Final[int] = 2100

_QUESTION_ID_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_UTC_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)


class MinimumModelCapability(str, Enum):
    """The smallest executor class permitted to answer the question."""

    LOCAL_SMALL_SPECIALIST = "local_small_specialist_model"
    LOCAL_OR_REMOTE_MEDIUM = "local_or_remote_medium_model"
    REMOTE_STRONG_OR_FRONTIER = "remote_strong_or_frontier_model"
    HUMAN_DECISION = "human_decision"


class AdmissibleDecision(str, Enum):
    """Closed route outcomes that an answer may affect.

    These values deliberately mirror the existing ``ModelRoute`` outcomes;
    this contract does not introduce another routing authority.
    """

    DETERMINISTIC_ONLY = "deterministic_only"
    SMALL_LOCAL_MODEL = "small_local_model"
    MEDIUM_MODEL = "medium_model"
    FRONTIER_MODEL = "frontier_model"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


def _text(value: Any, name: str, *, maximum: int) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise HarnessError(f"{name} must be a nonempty trimmed string")
    if value != unicodedata.normalize("NFC", value):
        raise HarnessError(f"{name} must be normalized as NFC")
    if any(not character.isprintable() for character in value):
        raise HarnessError(f"{name} contains non-printable characters")
    if len(value) > maximum:
        raise HarnessError(f"{name} must be at most {maximum} characters")
    return value


def _bounded_int(value: Any, name: str, *, maximum: int, minimum: int = 1) -> int:
    if type(value) is not int or isinstance(value, bool) or not minimum <= value <= maximum:
        raise HarnessError(f"{name} must be an integer from {minimum} through {maximum}")
    return value


def _canonical_texts(
    value: Any,
    name: str,
    *,
    maximum_items: int,
    maximum_text: int,
    minimum_items: int = 0,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise HarnessError(f"{name} must be a list")
    if not minimum_items <= len(value) <= maximum_items:
        raise HarnessError(
            f"{name} must contain from {minimum_items} through {maximum_items} items"
        )
    values = tuple(_text(item, name, maximum=maximum_text) for item in value)
    if len(values) != len(set(values)):
        raise HarnessError(f"{name} must not contain duplicates")
    return tuple(sorted(values))


def _decision_impacts(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise HarnessError("candidate_decisions_answer_could_change must be a list")
    # One outcome cannot demonstrate that an answer changes a choice.  A
    # question must name at least two admissible alternative outcomes.
    if not 2 <= len(value) <= len(AdmissibleDecision):
        raise HarnessError(
            "candidate_decisions_answer_could_change must name at least two "
            "admissible decisions"
        )
    try:
        decisions = tuple(AdmissibleDecision(item).value for item in value)
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            "candidate_decisions_answer_could_change contains a non-admissible decision"
        ) from exc
    if len(decisions) != len(set(decisions)):
        raise HarnessError("candidate_decisions_answer_could_change must not contain duplicates")
    return tuple(sorted(decisions))


def _response_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"type", "enum"}:
        raise HarnessError("response_schema must be the closed string-enum response schema")
    if value["type"] != "string":
        raise HarnessError("response_schema.type must be 'string'")
    answers = _canonical_texts(
        value["enum"],
        "response_schema.enum",
        maximum_items=MAX_RESPONSE_VALUES,
        maximum_text=MAX_RESPONSE_VALUE_CHARS,
        minimum_items=2,
    )
    return {"type": "string", "enum": list(answers)}


def _deadline(value: Any) -> str:
    text = _text(value, "deadline", maximum=20)
    if not _UTC_RE.fullmatch(text):
        raise HarnessError("deadline must be a canonical UTC RFC 3339 timestamp")
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise HarnessError("deadline must be a valid UTC timestamp") from exc
    if not MIN_DEADLINE_YEAR <= parsed.year <= MAX_DEADLINE_YEAR:
        raise HarnessError(
            f"deadline year must be from {MIN_DEADLINE_YEAR} through {MAX_DEADLINE_YEAR}"
        )
    return text


def canonical_question_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return canonical JSON bytes used for unresolved-question identity."""

    if not isinstance(payload, Mapping):
        raise HarnessError("unresolved question payload must be an object")
    try:
        return json.dumps(
            dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HarnessError("unresolved question payload is not canonical JSON") from exc


def question_identity_for(payload: Mapping[str, Any]) -> str:
    """Derive the closed identity from a validated non-identity question body."""

    body = dict(payload)
    body.pop("question_id", None)
    expected = UNRESOLVED_QUESTION_FIELDS - {"question_id"}
    if set(body) != expected:
        raise HarnessError("question identity requires every non-identity field")
    return "sha256:" + hashlib.sha256(canonical_question_bytes(body)).hexdigest()


@dataclass(frozen=True)
class UnresolvedQuestion:
    """A validated request for escalation after deterministic resolution failed."""

    question_id: str
    exact_question: str
    why_prior_deterministic_stages_could_not_resolve: str
    evidence_available: tuple[str, ...]
    evidence_missing: tuple[str, ...]
    candidate_decisions_answer_could_change: tuple[str, ...]
    minimum_model_capability: str
    context_budget: int
    response_schema: Mapping[str, Any]
    deadline: str
    cost_budget: int

    def __post_init__(self) -> None:
        normalized = _validate_payload(self.to_dict(), verify_identity=True)
        for name, value in normalized.items():
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "exact_question": self.exact_question,
            "why_prior_deterministic_stages_could_not_resolve": (
                self.why_prior_deterministic_stages_could_not_resolve
            ),
            "evidence_available": list(self.evidence_available),
            "evidence_missing": list(self.evidence_missing),
            "candidate_decisions_answer_could_change": list(
                self.candidate_decisions_answer_could_change
            ),
            "minimum_model_capability": self.minimum_model_capability,
            "context_budget": self.context_budget,
            "response_schema": dict(self.response_schema),
            "deadline": self.deadline,
            "cost_budget": self.cost_budget,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnresolvedQuestion":
        normalized = _validate_payload(payload, verify_identity=True)
        return cls(**normalized)

    def canonical_bytes(self) -> bytes:
        return canonical_question_bytes(self.to_dict())

    def round_trip(self) -> "UnresolvedQuestion":
        return self.from_dict(json.loads(self.canonical_bytes().decode("utf-8")))


def _validate_payload(payload: Mapping[str, Any], *, verify_identity: bool) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise HarnessError("UnresolvedQuestion must be an object")
    if set(payload) != UNRESOLVED_QUESTION_FIELDS:
        raise HarnessError(
            "UnresolvedQuestion fields must be exactly "
            f"{sorted(UNRESOLVED_QUESTION_FIELDS)}"
        )
    question_id = _text(payload["question_id"], "question_id", maximum=71)
    if not _QUESTION_ID_RE.fullmatch(question_id):
        raise HarnessError("question_id must be a sha256 identity")
    available = _canonical_texts(
        payload["evidence_available"],
        "evidence_available",
        maximum_items=MAX_EVIDENCE_ITEMS,
        maximum_text=MAX_EVIDENCE_CHARS,
    )
    missing = _canonical_texts(
        payload["evidence_missing"],
        "evidence_missing",
        maximum_items=MAX_EVIDENCE_ITEMS,
        maximum_text=MAX_EVIDENCE_CHARS,
        minimum_items=1,
    )
    if set(available).intersection(missing):
        raise HarnessError("evidence_available and evidence_missing must not overlap")
    try:
        capability = MinimumModelCapability(payload["minimum_model_capability"]).value
    except (TypeError, ValueError) as exc:
        raise HarnessError("minimum_model_capability is unsupported") from exc
    normalized = {
        "question_id": question_id,
        "exact_question": _text(
            payload["exact_question"], "exact_question", maximum=MAX_QUESTION_CHARS
        ),
        "why_prior_deterministic_stages_could_not_resolve": _text(
            payload["why_prior_deterministic_stages_could_not_resolve"],
            "why_prior_deterministic_stages_could_not_resolve",
            maximum=MAX_REASON_CHARS,
        ),
        "evidence_available": available,
        "evidence_missing": missing,
        "candidate_decisions_answer_could_change": _decision_impacts(
            payload["candidate_decisions_answer_could_change"]
        ),
        "minimum_model_capability": capability,
        "context_budget": _bounded_int(
            payload["context_budget"],
            "context_budget",
            maximum=MAX_CONTEXT_BUDGET,
        ),
        "response_schema": _response_schema(payload["response_schema"]),
        "deadline": _deadline(payload["deadline"]),
        "cost_budget": _bounded_int(
            payload["cost_budget"],
            "cost_budget",
            maximum=MAX_COST_BUDGET_MICROUSD,
        ),
    }
    if verify_identity and normalized["question_id"] != question_identity_for(normalized):
        raise HarnessError("question_id does not match the canonical unresolved question")
    return normalized


def build_unresolved_question(**fields: Any) -> UnresolvedQuestion:
    """Build a question and derive its sealed identity.

    Callers may supply ``question_id`` only when it equals the canonical
    identity.  This is useful when rehydrating externally stored records but
    prevents caller-selected identities.
    """

    provided = fields.get("question_id")
    body = dict(fields)
    body["question_id"] = "sha256:" + "0" * 64
    normalized = _validate_payload(body, verify_identity=False)
    identity = question_identity_for(normalized)
    if provided is not None and provided != identity:
        raise HarnessError("provided question_id does not match canonical question")
    normalized["question_id"] = identity
    return UnresolvedQuestion(**normalized)


def validate_unresolved_question(payload: Mapping[str, Any]) -> UnresolvedQuestion:
    """Fail-closed admission helper for untrusted unresolved-question data."""

    return UnresolvedQuestion.from_dict(payload)


def round_trip_unresolved_question(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a serialized unresolved-question record."""

    return validate_unresolved_question(payload).round_trip().to_dict()


def load_unresolved_question_schema() -> dict[str, Any]:
    """Load the colocated closed JSON Schema document without dispatching work."""

    try:
        schema = json.loads(UNRESOLVED_QUESTION_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HarnessError("unresolved-question schema document is unreadable") from exc
    if not isinstance(schema, dict) or schema.get("$id") != UNRESOLVED_QUESTION_SCHEMA:
        raise HarnessError("unresolved-question schema identity mismatch")
    if schema.get("additionalProperties") is not False:
        raise HarnessError("unresolved-question schema must be closed")
    return schema


__all__ = [
    "AdmissibleDecision",
    "MAX_CONTEXT_BUDGET",
    "MAX_COST_BUDGET_MICROUSD",
    "MAX_EVIDENCE_ITEMS",
    "MAX_RESPONSE_VALUES",
    "MinimumModelCapability",
    "UNRESOLVED_QUESTION_FIELDS",
    "UNRESOLVED_QUESTION_INTERFACE",
    "UNRESOLVED_QUESTION_SCHEMA",
    "UNRESOLVED_QUESTION_SCHEMA_PATH",
    "UnresolvedQuestion",
    "build_unresolved_question",
    "canonical_question_bytes",
    "load_unresolved_question_schema",
    "question_identity_for",
    "round_trip_unresolved_question",
    "validate_unresolved_question",
]
