"""BoundedExpertIteration@1 — generate-check-retain-refill-train-qualify.

The loop reuses the existing tactician curriculum projection, hammer/checker
authority lattice, and independent kernel gate.  Proposal roles never gain
proof authority.  Timeout is observational, never a falsehood label.  Only
independently checked verified successes and counterexamples enter
high-authority curriculum.  Hidden-test / holdout feedback is rejected.
A training or qualify receipt cannot self-promote a checkpoint.

Hard ceilings apply to candidates, search depth, provider calls, solver
time, rounds, and repeated examples.  Callers may tighten the ceilings but
must not raise them.  Durable checkpoints are content-addressed and resume
exactly from the last sealed stage.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final, Protocol

from .formal_verification_contracts import (
    AttemptStatus,
    ContractValidationError,
    content_identity,
)
from .goal_directed_tactician import (
    CurriculumAuthority,
    CurriculumClass,
    CurriculumProjection,
    ProofStateClass,
    ProofStateClassification,
    classify_proof_state,
    curriculum_authority_for,
    project_curriculum,
)
from .kernel_verification import KernelVerificationStatus
from .multi_prover_router import AttemptOutcome


BOUNDED_EXPERT_ITERATION_INTERFACE: Final = "BoundedExpertIteration@1"
BOUNDED_EXPERT_ITERATION_VERSION: Final = "1.0.0"
BOUNDED_EXPERT_ITERATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/bounded-expert-iteration@1"
)
EXPERT_ITERATION_BOUNDS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/expert-iteration-bounds@1"
)
EXPERT_ITERATION_ATTEMPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/expert-iteration-attempt@1"
)
EXPERT_ITERATION_ROUND_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/expert-iteration-round-receipt@1"
)
CURRICULUM_REVISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/expert-iteration-curriculum-revision@1"
)
EXPERT_ITERATION_CHECKPOINT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/expert-iteration-checkpoint@1"
)
EXPERT_ITERATION_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/expert-iteration-result@1"
)
EXPERT_ITERATION_REFILL_CANDIDATE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/expert-iteration-refill-candidate@1"
)

DEFAULT_STATE_FILENAME: Final = "bounded_expert_iteration.state.json"

# Hard safety ceilings.  Callers may tighten these but must not raise them.
MAX_CANDIDATES: Final = 8
MAX_DEPTH: Final = 8
MAX_CALLS: Final = 32
MAX_SOLVER_TIME_MS: Final = 30_000
MAX_ROUNDS: Final = 8
MAX_REPEATED_EXAMPLES: Final = 2
MAX_NO_PROGRESS_ROUNDS: Final = 3

ADMITTED_SPLITS: Final[frozenset[str]] = frozenset({"train", "development", "dev"})
HIDDEN_TEST_SPLITS: Final[frozenset[str]] = frozenset(
    {"test", "hidden", "hidden_test", "holdout", "eval", "evaluation"}
)
PROMOTION_AUTHORITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "current_checkpoint_pointer",
        "mutable_promotion_authority",
        "promotion",
        "promotion_pointer",
        "promotion_pointer_id",
        "self_promote",
        "self_promotion",
    }
)
HIDDEN_FEEDBACK_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "hidden_label",
        "hidden_labels",
        "hidden_test",
        "hidden_test_feedback",
        "holdout_feedback",
        "test_feedback",
        "test_set_metric",
    }
)

STAGE_ORDER: Final[tuple[str, ...]] = (
    "generate",
    "parse_type",
    "tactician",
    "hammer",
    "check",
    "retain",
    "refill",
    "train",
    "qualify",
)


class ExpertIterationError(ContractValidationError):
    """Malformed expert-iteration request, bound, or artifact."""


class ExpertIterationBoundError(ExpertIterationError):
    """A hard loop, call, depth, time, or repetition ceiling was reached."""


class HiddenTestFeedbackError(ExpertIterationError):
    """Hidden-test or holdout feedback was offered as a training signal."""


class UnverifiedRetentionError(ExpertIterationError):
    """An unverified success was offered as high-authority curriculum."""


class CheckpointSelfPromotionError(ExpertIterationError):
    """A checkpoint or qualify receipt tried to mutate promotion authority."""


class ExpertIterationStage(str, Enum):
    """Ordered stages of one expert-iteration round."""

    GENERATE = "generate"
    PARSE_TYPE = "parse_type"
    TACTICIAN = "tactician"
    HAMMER = "hammer"
    CHECK = "check"
    RETAIN = "retain"
    REFILL = "refill"
    TRAIN = "train"
    QUALIFY = "qualify"


class ExpertIterationOutcomeClass(str, Enum):
    """Closed mapping of terminal, timeout, and unavailable attempt classes."""

    VERIFIED_SUCCESS = "verified_success"
    CHECKED_COUNTEREXAMPLE = "checked_counterexample"
    PARSE_TYPE_FAILURE = "parse_type_failure"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    UNVERIFIED_SUCCESS = "unverified_success"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    HIDDEN_TEST_FEEDBACK = "hidden_test_feedback"
    EXHAUSTED = "exhausted"
    NO_PROGRESS = "no_progress"
    REPETITION_BOUNDED = "repetition_bounded"

    @property
    def retained_at_high_authority(self) -> bool:
        return self in {
            ExpertIterationOutcomeClass.VERIFIED_SUCCESS,
            ExpertIterationOutcomeClass.CHECKED_COUNTEREXAMPLE,
        }

    @property
    def timeout_is_falsehood(self) -> bool:
        return False

    @property
    def is_progress(self) -> bool:
        return self.retained_at_high_authority


class ExpertIterationStopReason(str, Enum):
    COMPLETED = "completed"
    EXHAUSTED = "exhausted"
    NO_PROGRESS = "no_progress"
    REPETITION_BOUNDED = "repetition_bounded"
    ROUND_BOUNDED = "round_bounded"
    CALL_BOUNDED = "call_bounded"
    CANDIDATE_BOUNDED = "candidate_bounded"
    DEPTH_BOUNDED = "depth_bounded"
    SOLVER_TIME_BOUNDED = "solver_time_bounded"
    HALTED = "halted"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ExpertIterationError(f"{name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise ExpertIterationError(f"{name} is required")
    if "\x00" in text:
        raise ExpertIterationError(f"{name} must not contain NUL")
    return text


def _optional_text(value: Any, name: str) -> str:
    return _text(value, name, required=False)


def _int(value: Any, name: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExpertIterationError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        raise ExpertIterationError(f"{name} is outside its bound")
    return value


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    if isinstance(value, kind):
        return value
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise ExpertIterationError(f"{name} is unsupported") from exc


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ExpertIterationError(f"{name} must be an object with string keys")
    return dict(value)


def _strings(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        items = value
    else:
        raise ExpertIterationError(f"{name} must be a sequence of strings")
    result: list[str] = []
    for item in items:
        text = _text(item, name)
        if text not in result:
            result.append(text)
    return tuple(result)


def _reject_promotion_authority(payload: Mapping[str, Any], *, noun: str) -> None:
    keys = {str(key).strip().casefold().replace("-", "_") for key in payload}
    if keys.intersection(PROMOTION_AUTHORITY_FIELDS):
        raise CheckpointSelfPromotionError(
            f"{noun} must not carry mutable promotion authority"
        )


def _reject_hidden_feedback(payload: Mapping[str, Any], *, noun: str) -> None:
    keys = {str(key).strip().casefold().replace("-", "_") for key in payload}
    if keys.intersection(HIDDEN_FEEDBACK_FIELDS):
        raise HiddenTestFeedbackError(f"{noun} cannot carry hidden-test feedback")
    nested = payload.get("metadata")
    if isinstance(nested, Mapping):
        nested_keys = {str(key).strip().casefold().replace("-", "_") for key in nested}
        if nested_keys.intersection(HIDDEN_FEEDBACK_FIELDS):
            raise HiddenTestFeedbackError(
                f"{noun} metadata cannot carry hidden-test feedback"
            )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "accepted", "proved"}
    return False


# ---------------------------------------------------------------------------
# Bounds, examples, attempts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertIterationBounds:
    """Closed numeric ceilings for one expert-iteration controller."""

    SCHEMA: ClassVar[str] = EXPERT_ITERATION_BOUNDS_SCHEMA

    max_candidates: int = MAX_CANDIDATES
    max_depth: int = MAX_DEPTH
    max_calls: int = MAX_CALLS
    max_solver_time_ms: int = MAX_SOLVER_TIME_MS
    max_rounds: int = MAX_ROUNDS
    max_repeated_examples: int = MAX_REPEATED_EXAMPLES
    max_no_progress_rounds: int = MAX_NO_PROGRESS_ROUNDS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_candidates",
            _int(self.max_candidates, "max_candidates", minimum=1, maximum=MAX_CANDIDATES),
        )
        object.__setattr__(
            self,
            "max_depth",
            _int(self.max_depth, "max_depth", minimum=1, maximum=MAX_DEPTH),
        )
        object.__setattr__(
            self,
            "max_calls",
            _int(self.max_calls, "max_calls", minimum=1, maximum=MAX_CALLS),
        )
        object.__setattr__(
            self,
            "max_solver_time_ms",
            _int(
                self.max_solver_time_ms,
                "max_solver_time_ms",
                minimum=1,
                maximum=MAX_SOLVER_TIME_MS,
            ),
        )
        object.__setattr__(
            self,
            "max_rounds",
            _int(self.max_rounds, "max_rounds", minimum=1, maximum=MAX_ROUNDS),
        )
        object.__setattr__(
            self,
            "max_repeated_examples",
            _int(
                self.max_repeated_examples,
                "max_repeated_examples",
                minimum=1,
                maximum=MAX_REPEATED_EXAMPLES,
            ),
        )
        object.__setattr__(
            self,
            "max_no_progress_rounds",
            _int(
                self.max_no_progress_rounds,
                "max_no_progress_rounds",
                minimum=1,
                maximum=MAX_NO_PROGRESS_ROUNDS,
            ),
        )

    @property
    def bounds_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": EXPERT_ITERATION_BOUNDS_SCHEMA,
            "max_candidates": self.max_candidates,
            "max_depth": self.max_depth,
            "max_calls": self.max_calls,
            "max_solver_time_ms": self.max_solver_time_ms,
            "max_rounds": self.max_rounds,
            "max_repeated_examples": self.max_repeated_examples,
            "max_no_progress_rounds": self.max_no_progress_rounds,
        }
        if include_id:
            payload["bounds_id"] = self.bounds_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ExpertIterationBounds":
        value = _mapping(payload, "bounds")
        _reject_promotion_authority(value, noun="expert-iteration bounds")
        return cls(
            max_candidates=int(value.get("max_candidates", MAX_CANDIDATES)),
            max_depth=int(value.get("max_depth", MAX_DEPTH)),
            max_calls=int(value.get("max_calls", MAX_CALLS)),
            max_solver_time_ms=int(value.get("max_solver_time_ms", MAX_SOLVER_TIME_MS)),
            max_rounds=int(value.get("max_rounds", MAX_ROUNDS)),
            max_repeated_examples=int(
                value.get("max_repeated_examples", MAX_REPEATED_EXAMPLES)
            ),
            max_no_progress_rounds=int(
                value.get("max_no_progress_rounds", MAX_NO_PROGRESS_ROUNDS)
            ),
        )


@dataclass(frozen=True)
class ExpertIterationExample:
    """One train/development example admitted into a round."""

    example_id: str
    obligation_id: str
    split: str
    statement: str = ""
    premises: tuple[str, ...] = ()
    fixture_kind: str = ""
    checkpoint_id: str = ""
    model_revision: str = ""
    tool_revision: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _text(self.example_id, "example_id"))
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id")
        )
        split = _text(self.split, "split").casefold()
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "statement", _optional_text(self.statement, "statement"))
        object.__setattr__(self, "premises", _strings(self.premises, "premises"))
        object.__setattr__(
            self, "fixture_kind", _optional_text(self.fixture_kind, "fixture_kind")
        )
        object.__setattr__(
            self, "checkpoint_id", _optional_text(self.checkpoint_id, "checkpoint_id")
        )
        object.__setattr__(
            self, "model_revision", _optional_text(self.model_revision, "model_revision")
        )
        object.__setattr__(
            self, "tool_revision", _optional_text(self.tool_revision, "tool_revision")
        )
        metadata = _mapping(self.metadata, "metadata")
        _reject_hidden_feedback(metadata, noun="example")
        _reject_promotion_authority(metadata, noun="example")
        object.__setattr__(self, "metadata", metadata)
        if split in HIDDEN_TEST_SPLITS:
            raise HiddenTestFeedbackError(
                "hidden-test, holdout, and evaluation splits cannot enter expert iteration"
            )
        if split not in ADMITTED_SPLITS:
            raise ExpertIterationError("split must be train or development")

    @property
    def progress_identity(self) -> str:
        return self.example_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "obligation_id": self.obligation_id,
            "split": self.split,
            "statement": self.statement,
            "premises": list(self.premises),
            "fixture_kind": self.fixture_kind,
            "checkpoint_id": self.checkpoint_id,
            "model_revision": self.model_revision,
            "tool_revision": self.tool_revision,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpertIterationExample":
        value = _mapping(payload, "example")
        return cls(
            example_id=str(value.get("example_id") or ""),
            obligation_id=str(value.get("obligation_id") or value.get("example_id") or ""),
            split=str(value.get("split") or "train"),
            statement=str(value.get("statement") or ""),
            premises=tuple(value.get("premises") or ()),
            fixture_kind=str(value.get("fixture_kind") or ""),
            checkpoint_id=str(value.get("checkpoint_id") or ""),
            model_revision=str(value.get("model_revision") or ""),
            tool_revision=str(value.get("tool_revision") or ""),
            metadata=value.get("metadata") or {},
        )


@dataclass(frozen=True)
class ExpertIterationAttempt:
    """Content-addressed attempt binding state, tools, resources, and outcome."""

    SCHEMA: ClassVar[str] = EXPERT_ITERATION_ATTEMPT_SCHEMA

    example_id: str
    obligation_id: str
    outcome_class: ExpertIterationOutcomeClass
    attempt_status: AttemptStatus
    stage: ExpertIterationStage
    independently_validated: bool
    kernel_verified: bool
    proof_authority: bool = False
    timeout_is_falsehood: bool = False
    depth: int = 0
    call_count: int = 0
    solver_time_ms: int = 0
    model_revision: str = ""
    tool_revision: str = ""
    checker_response: str = ""
    reason_code: str = ""
    curriculum_class: CurriculumClass = CurriculumClass.PARSE_TYPE
    curriculum_authority: CurriculumAuthority = CurriculumAuthority.NONE
    classification: ProofStateClassification | None = None
    projection: CurriculumProjection | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "example_id", _text(self.example_id, "example_id"))
        object.__setattr__(
            self, "obligation_id", _text(self.obligation_id, "obligation_id")
        )
        object.__setattr__(
            self,
            "outcome_class",
            _enum(self.outcome_class, ExpertIterationOutcomeClass, "outcome_class"),
        )
        object.__setattr__(
            self,
            "attempt_status",
            _enum(self.attempt_status, AttemptStatus, "attempt_status"),
        )
        object.__setattr__(self, "stage", _enum(self.stage, ExpertIterationStage, "stage"))
        object.__setattr__(self, "independently_validated", bool(self.independently_validated))
        object.__setattr__(self, "kernel_verified", bool(self.kernel_verified))
        object.__setattr__(self, "proof_authority", False)
        object.__setattr__(self, "timeout_is_falsehood", False)
        object.__setattr__(self, "depth", _int(self.depth, "depth"))
        object.__setattr__(self, "call_count", _int(self.call_count, "call_count"))
        object.__setattr__(
            self, "solver_time_ms", _int(self.solver_time_ms, "solver_time_ms")
        )
        object.__setattr__(
            self, "model_revision", _optional_text(self.model_revision, "model_revision")
        )
        object.__setattr__(
            self, "tool_revision", _optional_text(self.tool_revision, "tool_revision")
        )
        object.__setattr__(
            self,
            "checker_response",
            _optional_text(self.checker_response, "checker_response"),
        )
        object.__setattr__(self, "reason_code", _optional_text(self.reason_code, "reason_code"))
        if self.curriculum_class is not None:
            object.__setattr__(
                self,
                "curriculum_class",
                _enum(self.curriculum_class, CurriculumClass, "curriculum_class"),
            )
        if self.curriculum_authority is not None:
            object.__setattr__(
                self,
                "curriculum_authority",
                _enum(
                    self.curriculum_authority,
                    CurriculumAuthority,
                    "curriculum_authority",
                ),
            )
        metadata = _mapping(self.metadata, "metadata")
        _reject_hidden_feedback(metadata, noun="attempt")
        _reject_promotion_authority(metadata, noun="attempt")
        object.__setattr__(self, "metadata", metadata)
        if (
            self.curriculum_authority is CurriculumAuthority.HIGH
            and not self.outcome_class.retained_at_high_authority
        ):
            raise UnverifiedRetentionError(
                "high curriculum authority requires a verified success or checked counterexample"
            )
        if (
            self.curriculum_authority is CurriculumAuthority.HIGH
            and not self.independently_validated
        ):
            raise UnverifiedRetentionError(
                "high curriculum authority requires independent validation"
            )

    @property
    def attempt_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    @property
    def retained(self) -> bool:
        return (
            self.curriculum_authority is CurriculumAuthority.HIGH
            and self.outcome_class.retained_at_high_authority
        )

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": EXPERT_ITERATION_ATTEMPT_SCHEMA,
            "example_id": self.example_id,
            "obligation_id": self.obligation_id,
            "outcome_class": self.outcome_class.value,
            "attempt_status": self.attempt_status.value,
            "stage": self.stage.value,
            "independently_validated": self.independently_validated,
            "kernel_verified": self.kernel_verified,
            "proof_authority": False,
            "timeout_is_falsehood": False,
            "depth": self.depth,
            "call_count": self.call_count,
            "solver_time_ms": self.solver_time_ms,
            "model_revision": self.model_revision,
            "tool_revision": self.tool_revision,
            "checker_response": self.checker_response,
            "reason_code": self.reason_code,
            "curriculum_class": self.curriculum_class.value,
            "curriculum_authority": self.curriculum_authority.value,
            "retained": self.retained,
            "classification": (
                self.classification.to_dict() if self.classification is not None else None
            ),
            "projection": (
                self.projection.to_dict() if self.projection is not None else None
            ),
            "metadata": dict(self.metadata),
        }
        if include_id:
            payload["attempt_id"] = self.attempt_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpertIterationAttempt":
        value = _mapping(payload, "attempt")
        classification = value.get("classification")
        projection = value.get("projection")
        return cls(
            example_id=str(value.get("example_id") or ""),
            obligation_id=str(value.get("obligation_id") or ""),
            outcome_class=value.get(
                "outcome_class", ExpertIterationOutcomeClass.INCONCLUSIVE
            ),
            attempt_status=value.get("attempt_status", AttemptStatus.FAILED),
            stage=value.get("stage", ExpertIterationStage.CHECK),
            independently_validated=bool(value.get("independently_validated", False)),
            kernel_verified=bool(value.get("kernel_verified", False)),
            depth=int(value.get("depth") or 0),
            call_count=int(value.get("call_count") or 0),
            solver_time_ms=int(value.get("solver_time_ms") or 0),
            model_revision=str(value.get("model_revision") or ""),
            tool_revision=str(value.get("tool_revision") or ""),
            checker_response=str(value.get("checker_response") or ""),
            reason_code=str(value.get("reason_code") or ""),
            curriculum_class=value.get("curriculum_class", CurriculumClass.PARSE_TYPE),
            curriculum_authority=value.get(
                "curriculum_authority", CurriculumAuthority.NONE
            ),
            classification=(
                ProofStateClassification.from_dict(classification)
                if isinstance(classification, Mapping)
                else classification
            ),
            projection=(
                CurriculumProjection.from_dict(projection)
                if isinstance(projection, Mapping)
                else projection
            ),
            metadata=value.get("metadata") or {},
        )


# ---------------------------------------------------------------------------
# Outcome mapping
# ---------------------------------------------------------------------------


_TIMEOUT_MARKERS: Final = frozenset(
    {"timeout", "timed_out", "time_out", KernelVerificationStatus.TIMED_OUT.value}
)
_UNAVAILABLE_MARKERS: Final = frozenset(
    {
        "unavailable",
        "outage",
        "provider_outage",
        KernelVerificationStatus.UNAVAILABLE.value,
        AttemptOutcome.UNAVAILABLE.value,
        AttemptStatus.UNAVAILABLE.value,
    }
)
_UNSUPPORTED_MARKERS: Final = frozenset(
    {
        "unsupported",
        AttemptOutcome.UNSUPPORTED.value,
        AttemptStatus.UNSUPPORTED.value,
    }
)
_PARSE_MARKERS: Final = frozenset(
    {"parse_error", "parse_failed", "type_error", "elaboration_failed"}
)


def _token(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().casefold()


def map_attempt_outcome(payload: Mapping[str, Any] | None) -> ExpertIterationOutcomeClass:
    """Map one stage payload onto the closed terminal/timeout/unavailable classes."""

    value = _mapping(payload, "attempt outcome")
    split = _token(value.get("split"))
    keys = {str(key).strip().casefold().replace("-", "_") for key in value}
    if (
        split in HIDDEN_TEST_SPLITS
        or _truthy(value.get("hidden_test_feedback"))
        or keys.intersection(HIDDEN_FEEDBACK_FIELDS)
    ):
        return ExpertIterationOutcomeClass.HIDDEN_TEST_FEEDBACK

    tokens = {
        _token(value.get("status")),
        _token(value.get("outcome")),
        _token(value.get("kind")),
        _token(value.get("failure_code")),
        _token(value.get("reason_code")),
        _token(value.get("kernel_status")),
        _token(value.get("attempt_outcome")),
    }
    if isinstance(value.get("timeout"), Mapping) or _truthy(value.get("timeout")):
        tokens.add("timeout")
    kernel = value.get("kernel_outcome")
    if isinstance(kernel, Mapping):
        tokens.add(_token(kernel.get("status")))
        tokens.add(_token(kernel.get("outcome")))
    parse = value.get("parse_outcome")
    if isinstance(parse, Mapping):
        tokens.add(_token(parse.get("status")))
    elab = value.get("elaboration_outcome")
    if isinstance(elab, Mapping):
        tokens.add(_token(elab.get("status")))

    if tokens & _TIMEOUT_MARKERS:
        return ExpertIterationOutcomeClass.TIMEOUT
    if tokens & _UNAVAILABLE_MARKERS:
        return ExpertIterationOutcomeClass.UNAVAILABLE
    if tokens & _UNSUPPORTED_MARKERS:
        return ExpertIterationOutcomeClass.UNSUPPORTED
    if tokens & _PARSE_MARKERS:
        return ExpertIterationOutcomeClass.PARSE_TYPE_FAILURE

    independently = bool(value.get("independently_validated"))
    kernel_verified = bool(value.get("kernel_verified"))
    if _token(value.get("kernel_status")) in {"accepted", "proved"} and independently:
        kernel_verified = True
    has_cex = bool(
        value.get("counterexample")
        or value.get("counterexamples")
        or _token(value.get("outcome")) == "counterexample"
        or _token(value.get("kind")) == "counterexample"
    )
    claimed_success = _token(value.get("claimed")) in {"proved", "success", "verified"}
    if has_cex and independently:
        return ExpertIterationOutcomeClass.CHECKED_COUNTEREXAMPLE
    if has_cex:
        return ExpertIterationOutcomeClass.INCONCLUSIVE
    if kernel_verified and independently:
        return ExpertIterationOutcomeClass.VERIFIED_SUCCESS
    if claimed_success or _token(value.get("claimed")) == "proved":
        return ExpertIterationOutcomeClass.UNVERIFIED_SUCCESS
    if _token(value.get("status")) == "rejected" or _token(value.get("reason_code")) == "kernel_rejected":
        return ExpertIterationOutcomeClass.REJECTED
    return ExpertIterationOutcomeClass.INCONCLUSIVE


def outcome_to_attempt_status(
    outcome: ExpertIterationOutcomeClass,
) -> AttemptStatus:
    if outcome is ExpertIterationOutcomeClass.VERIFIED_SUCCESS:
        return AttemptStatus.SUCCEEDED
    if outcome is ExpertIterationOutcomeClass.CHECKED_COUNTEREXAMPLE:
        return AttemptStatus.SUCCEEDED
    if outcome is ExpertIterationOutcomeClass.TIMEOUT:
        return AttemptStatus.TIMED_OUT
    if outcome is ExpertIterationOutcomeClass.UNAVAILABLE:
        return AttemptStatus.UNAVAILABLE
    if outcome is ExpertIterationOutcomeClass.UNSUPPORTED:
        return AttemptStatus.UNSUPPORTED
    if outcome in {
        ExpertIterationOutcomeClass.EXHAUSTED,
        ExpertIterationOutcomeClass.NO_PROGRESS,
        ExpertIterationOutcomeClass.REPETITION_BOUNDED,
    }:
        return AttemptStatus.BLOCKED
    return AttemptStatus.FAILED


def outcome_to_classification(
    outcome: ExpertIterationOutcomeClass,
    *,
    independently_validated: bool,
    kernel_verified: bool,
    reason_code: str = "",
) -> ProofStateClassification:
    if outcome is ExpertIterationOutcomeClass.TIMEOUT:
        return ProofStateClassification(
            state_class=ProofStateClass.TIMEOUT,
            curriculum_class=CurriculumClass.TIMEOUT,
            independently_validated=independently_validated,
            kernel_verified=False,
            reason_code=reason_code or "timeout_is_not_falsehood",
        )
    if outcome is ExpertIterationOutcomeClass.CHECKED_COUNTEREXAMPLE:
        return ProofStateClassification(
            state_class=ProofStateClass.COUNTEREXAMPLE,
            curriculum_class=CurriculumClass.COUNTEREXAMPLE,
            independently_validated=independently_validated,
            kernel_verified=kernel_verified,
            reason_code=reason_code or "checked_counterexample",
        )
    if outcome is ExpertIterationOutcomeClass.VERIFIED_SUCCESS:
        return ProofStateClassification(
            state_class=ProofStateClass.CLOSED,
            curriculum_class=CurriculumClass.VERIFIED_SUCCESS,
            independently_validated=independently_validated,
            kernel_verified=kernel_verified,
            reason_code=reason_code or "independently_validated_kernel_success",
        )
    if outcome is ExpertIterationOutcomeClass.PARSE_TYPE_FAILURE:
        return ProofStateClassification(
            state_class=ProofStateClass.PARSE_ERROR,
            curriculum_class=CurriculumClass.PARSE_TYPE,
            independently_validated=independently_validated,
            kernel_verified=False,
            reason_code=reason_code or "parse_type_failure",
        )
    if outcome is ExpertIterationOutcomeClass.UNAVAILABLE:
        return ProofStateClassification(
            state_class=ProofStateClass.STUCK,
            curriculum_class=CurriculumClass.PARSE_TYPE,
            independently_validated=independently_validated,
            kernel_verified=False,
            reason_code=reason_code or "unavailable_is_not_falsehood",
        )
    return ProofStateClassification(
        state_class=ProofStateClass.OPEN,
        curriculum_class=CurriculumClass.PARSE_TYPE,
        independently_validated=independently_validated,
        kernel_verified=False,
        reason_code=reason_code or outcome.value,
    )


def build_attempt(
    example: ExpertIterationExample,
    payload: Mapping[str, Any] | None,
    *,
    stage: ExpertIterationStage,
    depth: int = 0,
    call_count: int = 0,
    solver_time_ms: int = 0,
) -> ExpertIterationAttempt:
    """Classify one stage payload and project it into a curriculum attempt."""

    value = _mapping(payload, "stage payload")
    merged = {
        "split": example.split,
        "independently_validated": bool(value.get("independently_validated", False)),
        "kernel_verified": bool(value.get("kernel_verified", False)),
        **value,
    }
    outcome = map_attempt_outcome(merged)
    independently = bool(merged.get("independently_validated"))
    kernel_verified = bool(merged.get("kernel_verified"))
    if outcome is ExpertIterationOutcomeClass.VERIFIED_SUCCESS:
        independently = True
        kernel_verified = True
    if outcome is ExpertIterationOutcomeClass.CHECKED_COUNTEREXAMPLE:
        independently = True
    classification = outcome_to_classification(
        outcome,
        independently_validated=independently,
        kernel_verified=kernel_verified,
        reason_code=str(merged.get("reason_code") or ""),
    )
    if not classification.reason_code:
        classification = classify_proof_state(
            merged,
            independently_validated=independently,
            kernel_verified=kernel_verified,
        )
        if outcome is ExpertIterationOutcomeClass.TIMEOUT:
            classification = outcome_to_classification(
                outcome,
                independently_validated=independently,
                kernel_verified=False,
            )
    projection = project_curriculum(
        classification,
        independently_validated=independently,
    )
    if (
        outcome is ExpertIterationOutcomeClass.UNVERIFIED_SUCCESS
        and projection.authority is CurriculumAuthority.HIGH
    ):
        raise UnverifiedRetentionError("unverified success cannot upgrade curriculum")
    return ExpertIterationAttempt(
        example_id=example.example_id,
        obligation_id=example.obligation_id,
        outcome_class=outcome,
        attempt_status=outcome_to_attempt_status(outcome),
        stage=stage,
        independently_validated=independently,
        kernel_verified=kernel_verified
        if outcome is ExpertIterationOutcomeClass.VERIFIED_SUCCESS
        else bool(classification.kernel_verified),
        depth=depth,
        call_count=call_count,
        solver_time_ms=solver_time_ms,
        model_revision=example.model_revision or str(merged.get("model_revision") or ""),
        tool_revision=example.tool_revision or str(merged.get("tool_revision") or ""),
        checker_response=str(merged.get("checker_response") or ""),
        reason_code=projection.reason_code or classification.reason_code or outcome.value,
        curriculum_class=projection.curriculum_class,
        curriculum_authority=projection.authority,
        classification=classification,
        projection=projection,
        metadata={"fixture_kind": example.fixture_kind} if example.fixture_kind else {},
    )


# ---------------------------------------------------------------------------
# Curriculum revision, receipts, checkpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurriculumRevision:
    """Content-addressed set of high-authority retained traces."""

    SCHEMA: ClassVar[str] = CURRICULUM_REVISION_SCHEMA

    campaign_id: str
    round_index: int
    retained_attempt_ids: tuple[str, ...] = ()
    residual_example_ids: tuple[str, ...] = ()
    parent_revision_id: str = ""
    checkpoint_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign_id", _text(self.campaign_id, "campaign_id"))
        object.__setattr__(
            self, "round_index", _int(self.round_index, "round_index", minimum=0)
        )
        object.__setattr__(
            self,
            "retained_attempt_ids",
            _strings(self.retained_attempt_ids, "retained_attempt_ids"),
        )
        object.__setattr__(
            self,
            "residual_example_ids",
            _strings(self.residual_example_ids, "residual_example_ids"),
        )
        object.__setattr__(
            self,
            "parent_revision_id",
            _optional_text(self.parent_revision_id, "parent_revision_id"),
        )
        object.__setattr__(
            self, "checkpoint_id", _optional_text(self.checkpoint_id, "checkpoint_id")
        )
        metadata = _mapping(self.metadata, "metadata")
        _reject_promotion_authority(metadata, noun="curriculum revision")
        _reject_hidden_feedback(metadata, noun="curriculum revision")
        object.__setattr__(self, "metadata", metadata)

    @property
    def revision_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    @property
    def result_identity(self) -> str:
        return self.revision_id

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CURRICULUM_REVISION_SCHEMA,
            "campaign_id": self.campaign_id,
            "round_index": self.round_index,
            "retained_attempt_ids": list(self.retained_attempt_ids),
            "residual_example_ids": list(self.residual_example_ids),
            "parent_revision_id": self.parent_revision_id,
            "checkpoint_id": self.checkpoint_id,
            "metadata": dict(self.metadata),
        }
        if include_id:
            payload["revision_id"] = self.revision_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CurriculumRevision":
        value = _mapping(payload, "curriculum revision")
        return cls(
            campaign_id=str(value.get("campaign_id") or ""),
            round_index=int(value.get("round_index") or 0),
            retained_attempt_ids=tuple(value.get("retained_attempt_ids") or ()),
            residual_example_ids=tuple(value.get("residual_example_ids") or ()),
            parent_revision_id=str(value.get("parent_revision_id") or ""),
            checkpoint_id=str(value.get("checkpoint_id") or ""),
            metadata=value.get("metadata") or {},
        )


@dataclass(frozen=True)
class ExpertIterationRoundReceipt:
    """Content-addressed receipt for one sealed expert-iteration round."""

    SCHEMA: ClassVar[str] = EXPERT_ITERATION_ROUND_RECEIPT_SCHEMA

    campaign_id: str
    round_index: int
    bounds_id: str
    curriculum_revision_id: str
    outcome_classes: tuple[str, ...] = ()
    retained_count: int = 0
    residual_count: int = 0
    call_count: int = 0
    solver_time_ms: int = 0
    no_progress_streak: int = 0
    stop_reason: ExpertIterationStopReason = ExpertIterationStopReason.COMPLETED
    refill_disposition: str = ""
    promotion_authority: bool = False
    attempt_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign_id", _text(self.campaign_id, "campaign_id"))
        object.__setattr__(
            self, "round_index", _int(self.round_index, "round_index", minimum=0)
        )
        object.__setattr__(self, "bounds_id", _text(self.bounds_id, "bounds_id"))
        object.__setattr__(
            self,
            "curriculum_revision_id",
            _text(self.curriculum_revision_id, "curriculum_revision_id"),
        )
        object.__setattr__(
            self, "outcome_classes", _strings(self.outcome_classes, "outcome_classes")
        )
        for name in (
            "retained_count",
            "residual_count",
            "call_count",
            "solver_time_ms",
            "no_progress_streak",
        ):
            object.__setattr__(self, name, _int(getattr(self, name), name))
        object.__setattr__(
            self,
            "stop_reason",
            _enum(self.stop_reason, ExpertIterationStopReason, "stop_reason"),
        )
        object.__setattr__(
            self,
            "refill_disposition",
            _optional_text(self.refill_disposition, "refill_disposition"),
        )
        object.__setattr__(self, "promotion_authority", False)
        object.__setattr__(self, "attempt_ids", _strings(self.attempt_ids, "attempt_ids"))
        metadata = _mapping(self.metadata, "metadata")
        _reject_promotion_authority(metadata, noun="round receipt")
        object.__setattr__(self, "metadata", metadata)

    @property
    def receipt_id(self) -> str:
        return content_identity(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": EXPERT_ITERATION_ROUND_RECEIPT_SCHEMA,
            "campaign_id": self.campaign_id,
            "round_index": self.round_index,
            "bounds_id": self.bounds_id,
            "curriculum_revision_id": self.curriculum_revision_id,
            "outcome_classes": list(self.outcome_classes),
            "retained_count": self.retained_count,
            "residual_count": self.residual_count,
            "call_count": self.call_count,
            "solver_time_ms": self.solver_time_ms,
            "no_progress_streak": self.no_progress_streak,
            "stop_reason": self.stop_reason.value,
            "refill_disposition": self.refill_disposition,
            "promotion_authority": False,
            "attempt_ids": list(self.attempt_ids),
            "metadata": dict(self.metadata),
        }
        if include_id:
            payload["receipt_id"] = self.receipt_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpertIterationRoundReceipt":
        value = _mapping(payload, "round receipt")
        _reject_promotion_authority(value, noun="round receipt")
        return cls(
            campaign_id=str(value.get("campaign_id") or ""),
            round_index=int(value.get("round_index") or 0),
            bounds_id=str(value.get("bounds_id") or ""),
            curriculum_revision_id=str(value.get("curriculum_revision_id") or ""),
            outcome_classes=tuple(value.get("outcome_classes") or ()),
            retained_count=int(value.get("retained_count") or 0),
            residual_count=int(value.get("residual_count") or 0),
            call_count=int(value.get("call_count") or 0),
            solver_time_ms=int(value.get("solver_time_ms") or 0),
            no_progress_streak=int(value.get("no_progress_streak") or 0),
            stop_reason=value.get("stop_reason", ExpertIterationStopReason.COMPLETED),
            refill_disposition=str(value.get("refill_disposition") or ""),
            attempt_ids=tuple(value.get("attempt_ids") or ()),
            metadata=value.get("metadata") or {},
        )


@dataclass(frozen=True)
class ExpertIterationRefillCandidate:
    """Residual offered to the existing campaign refill control surface."""

    SCHEMA: ClassVar[str] = EXPERT_ITERATION_REFILL_CANDIDATE_SCHEMA

    candidate_id: str
    trigger: str
    residual_count: int = 1
    curriculum_key: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "trigger", _text(self.trigger, "trigger"))
        object.__setattr__(
            self, "residual_count", _int(self.residual_count, "residual_count", minimum=1)
        )
        object.__setattr__(
            self, "curriculum_key", _optional_text(self.curriculum_key, "curriculum_key")
        )
        metadata = _mapping(self.metadata, "metadata")
        _reject_hidden_feedback(metadata, noun="refill candidate")
        object.__setattr__(self, "metadata", metadata)

    @property
    def progress_identity(self) -> str:
        return self.curriculum_key or self.candidate_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXPERT_ITERATION_REFILL_CANDIDATE_SCHEMA,
            "candidate_id": self.candidate_id,
            "trigger": self.trigger,
            "residual_count": self.residual_count,
            "curriculum_key": self.curriculum_key,
            "progress_identity": self.progress_identity,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExpertIterationCheckpoint:
    """Exact durable resume state. Never a promotion pointer."""

    SCHEMA: ClassVar[str] = EXPERT_ITERATION_CHECKPOINT_SCHEMA

    campaign_id: str
    bounds: ExpertIterationBounds
    fence: int = 0
    next_round_index: int = 0
    sealed_stage: str = ""
    no_progress_streak: int = 0
    last_progress_identity: str = ""
    call_count: int = 0
    solver_time_ms: int = 0
    curriculum_repetitions: Mapping[str, int] = field(default_factory=dict)
    completed_example_ids: tuple[str, ...] = ()
    pending_examples: tuple[Mapping[str, Any], ...] = ()
    retained_attempt_ids: tuple[str, ...] = ()
    residual_example_ids: tuple[str, ...] = ()
    attempt_payloads: tuple[Mapping[str, Any], ...] = ()
    receipt_payloads: tuple[Mapping[str, Any], ...] = ()
    revision_payloads: tuple[Mapping[str, Any], ...] = ()
    refill_payloads: tuple[Mapping[str, Any], ...] = ()
    curriculum_revision_ids: tuple[str, ...] = ()
    parent_revision_id: str = ""
    checkpoint_id: str = ""
    promotion_authority: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign_id", _text(self.campaign_id, "campaign_id"))
        bounds = (
            self.bounds
            if isinstance(self.bounds, ExpertIterationBounds)
            else ExpertIterationBounds.from_dict(self.bounds)
        )
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "fence", _int(self.fence, "fence"))
        object.__setattr__(
            self, "next_round_index", _int(self.next_round_index, "next_round_index")
        )
        object.__setattr__(
            self, "sealed_stage", _optional_text(self.sealed_stage, "sealed_stage")
        )
        object.__setattr__(
            self, "no_progress_streak", _int(self.no_progress_streak, "no_progress_streak")
        )
        object.__setattr__(
            self,
            "last_progress_identity",
            _optional_text(self.last_progress_identity, "last_progress_identity"),
        )
        object.__setattr__(self, "call_count", _int(self.call_count, "call_count"))
        object.__setattr__(
            self, "solver_time_ms", _int(self.solver_time_ms, "solver_time_ms")
        )
        repetitions = {
            _text(key, "curriculum key"): _int(value, "curriculum repetition")
            for key, value in dict(self.curriculum_repetitions).items()
        }
        object.__setattr__(self, "curriculum_repetitions", repetitions)
        object.__setattr__(
            self,
            "completed_example_ids",
            _strings(self.completed_example_ids, "completed_example_ids"),
        )
        pending: list[dict[str, Any]] = []
        for item in self.pending_examples:
            pending.append(dict(_mapping(item, "pending example")))
        object.__setattr__(self, "pending_examples", tuple(pending))
        object.__setattr__(
            self,
            "retained_attempt_ids",
            _strings(self.retained_attempt_ids, "retained_attempt_ids"),
        )
        object.__setattr__(
            self,
            "residual_example_ids",
            _strings(self.residual_example_ids, "residual_example_ids"),
        )
        attempts = tuple(dict(_mapping(item, "attempt")) for item in self.attempt_payloads)
        object.__setattr__(self, "attempt_payloads", attempts)
        receipts = tuple(
            dict(_mapping(item, "receipt")) for item in self.receipt_payloads
        )
        object.__setattr__(self, "receipt_payloads", receipts)
        object.__setattr__(
            self,
            "revision_payloads",
            tuple(dict(_mapping(item, "revision")) for item in self.revision_payloads),
        )
        object.__setattr__(
            self,
            "refill_payloads",
            tuple(dict(_mapping(item, "refill")) for item in self.refill_payloads),
        )
        object.__setattr__(
            self,
            "curriculum_revision_ids",
            _strings(self.curriculum_revision_ids, "curriculum_revision_ids"),
        )
        object.__setattr__(
            self,
            "parent_revision_id",
            _optional_text(self.parent_revision_id, "parent_revision_id"),
        )
        object.__setattr__(
            self, "checkpoint_id", _optional_text(self.checkpoint_id, "checkpoint_id")
        )
        object.__setattr__(self, "promotion_authority", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXPERT_ITERATION_CHECKPOINT_SCHEMA,
            "interface": BOUNDED_EXPERT_ITERATION_INTERFACE,
            "campaign_id": self.campaign_id,
            "bounds": self.bounds.to_dict(),
            "fence": self.fence,
            "next_round_index": self.next_round_index,
            "sealed_stage": self.sealed_stage,
            "no_progress_streak": self.no_progress_streak,
            "last_progress_identity": self.last_progress_identity,
            "call_count": self.call_count,
            "solver_time_ms": self.solver_time_ms,
            "curriculum_repetitions": dict(self.curriculum_repetitions),
            "completed_example_ids": list(self.completed_example_ids),
            "pending_examples": [dict(item) for item in self.pending_examples],
            "retained_attempt_ids": list(self.retained_attempt_ids),
            "residual_example_ids": list(self.residual_example_ids),
            "attempt_payloads": [dict(item) for item in self.attempt_payloads],
            "receipt_payloads": [dict(item) for item in self.receipt_payloads],
            "revision_payloads": [dict(item) for item in self.revision_payloads],
            "refill_payloads": [dict(item) for item in self.refill_payloads],
            "curriculum_revision_ids": list(self.curriculum_revision_ids),
            "parent_revision_id": self.parent_revision_id,
            "checkpoint_id": self.checkpoint_id,
            "promotion_authority": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExpertIterationCheckpoint":
        value = _mapping(payload, "checkpoint")
        _reject_promotion_authority(value, noun="expert-iteration checkpoint")
        return cls(
            campaign_id=str(value.get("campaign_id") or ""),
            bounds=ExpertIterationBounds.from_dict(value.get("bounds") or {}),
            fence=int(value.get("fence") or 0),
            next_round_index=int(value.get("next_round_index") or 0),
            sealed_stage=str(value.get("sealed_stage") or ""),
            no_progress_streak=int(value.get("no_progress_streak") or 0),
            last_progress_identity=str(value.get("last_progress_identity") or ""),
            call_count=int(value.get("call_count") or 0),
            solver_time_ms=int(value.get("solver_time_ms") or 0),
            curriculum_repetitions=value.get("curriculum_repetitions") or {},
            completed_example_ids=tuple(value.get("completed_example_ids") or ()),
            pending_examples=tuple(value.get("pending_examples") or ()),
            retained_attempt_ids=tuple(value.get("retained_attempt_ids") or ()),
            residual_example_ids=tuple(value.get("residual_example_ids") or ()),
            attempt_payloads=tuple(value.get("attempt_payloads") or ()),
            receipt_payloads=tuple(value.get("receipt_payloads") or ()),
            revision_payloads=tuple(value.get("revision_payloads") or ()),
            refill_payloads=tuple(value.get("refill_payloads") or ()),
            curriculum_revision_ids=tuple(value.get("curriculum_revision_ids") or ()),
            parent_revision_id=str(value.get("parent_revision_id") or ""),
            checkpoint_id=str(value.get("checkpoint_id") or ""),
        )

    def write(self, path: Path) -> None:
        _atomic_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path | str) -> "ExpertIterationCheckpoint":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ExpertIterationError("checkpoint payload must be an object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class ExpertIterationResult:
    """Terminal projection of one bounded expert-iteration run."""

    SCHEMA: ClassVar[str] = EXPERT_ITERATION_RESULT_SCHEMA

    campaign_id: str
    complete: bool
    stop_reason: ExpertIterationStopReason
    bounds: ExpertIterationBounds
    attempts: tuple[ExpertIterationAttempt, ...] = ()
    receipts: tuple[ExpertIterationRoundReceipt, ...] = ()
    curriculum_revisions: tuple[CurriculumRevision, ...] = ()
    refill_candidates: tuple[ExpertIterationRefillCandidate, ...] = ()
    no_progress_streak: int = 0
    call_count: int = 0
    solver_time_ms: int = 0
    promotion_authority: bool = False
    fence: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign_id", _text(self.campaign_id, "campaign_id"))
        object.__setattr__(self, "complete", bool(self.complete))
        object.__setattr__(
            self,
            "stop_reason",
            _enum(self.stop_reason, ExpertIterationStopReason, "stop_reason"),
        )
        object.__setattr__(self, "promotion_authority", False)

    @property
    def latest_curriculum_revision(self) -> CurriculumRevision | None:
        if not self.curriculum_revisions:
            return None
        return self.curriculum_revisions[-1]

    @property
    def retained_attempts(self) -> tuple[ExpertIterationAttempt, ...]:
        return tuple(item for item in self.attempts if item.retained)

    def to_dict(self) -> dict[str, Any]:
        latest = self.latest_curriculum_revision
        return {
            "schema": EXPERT_ITERATION_RESULT_SCHEMA,
            "interface": BOUNDED_EXPERT_ITERATION_INTERFACE,
            "campaign_id": self.campaign_id,
            "complete": self.complete,
            "stop_reason": self.stop_reason.value,
            "bounds": self.bounds.to_dict(),
            "attempts": [item.to_dict() for item in self.attempts],
            "receipts": [item.to_dict() for item in self.receipts],
            "curriculum_revisions": [item.to_dict() for item in self.curriculum_revisions],
            "refill_candidates": [item.to_dict() for item in self.refill_candidates],
            "no_progress_streak": self.no_progress_streak,
            "call_count": self.call_count,
            "solver_time_ms": self.solver_time_ms,
            "promotion_authority": False,
            "fence": self.fence,
            "curriculum_revision_id": latest.revision_id if latest is not None else "",
            "result_identity": latest.revision_id if latest is not None else "",
        }


# ---------------------------------------------------------------------------
# Compact fixture recipes
# ---------------------------------------------------------------------------


EXPERT_ITERATION_FIXTURE_KINDS: Final[tuple[str, ...]] = (
    "verified_success",
    "checked_counterexample",
    "parse_type_failure",
    "timeout",
    "unavailable",
    "unsupported",
    "unverified_success",
    "hidden_test_feedback",
    "rejected",
    "inconclusive",
)


def expert_iteration_fixture_payload(kind: str) -> dict[str, Any]:
    """Compact recipe for one closed outcome class. Not a golden envelope."""

    selected = _text(kind, "fixture kind").casefold()
    recipes: dict[str, dict[str, Any]] = {
        "verified_success": {
            "status": "succeeded",
            "independently_validated": True,
            "kernel_verified": True,
            "kernel_status": "accepted",
            "kernel_outcome": {"status": "accepted"},
            "parse_outcome": {"status": "parsed"},
            "elaboration_outcome": {"status": "typed"},
            "checker_response": "kernel:accepted",
        },
        "checked_counterexample": {
            "status": "succeeded",
            "independently_validated": True,
            "counterexample": True,
            "outcome": "counterexample",
            "checker_response": "solver:counterexample",
        },
        "parse_type_failure": {
            "status": "failed",
            "parse_outcome": {"status": "parse_error"},
            "reason_code": "parse_error",
        },
        "timeout": {
            "status": "timed_out",
            "timeout": True,
            "reason_code": "timeout_is_not_falsehood",
        },
        "unavailable": {
            "status": "unavailable",
            "reason_code": "provider_outage",
        },
        "unsupported": {
            "status": "unsupported",
            "reason_code": "explicit_fallback_required",
        },
        "unverified_success": {
            "status": "succeeded",
            "claimed": "proved",
            "independently_validated": False,
            "kernel_verified": False,
        },
        "hidden_test_feedback": {
            "split": "hidden",
            "hidden_test_feedback": True,
            "status": "succeeded",
        },
        "rejected": {
            "status": "rejected",
            "kernel_outcome": {"status": "rejected"},
            "reason_code": "kernel_rejected",
        },
        "inconclusive": {
            "status": "inconclusive",
            "reason_code": "inconclusive",
        },
    }
    if selected not in recipes:
        raise ExpertIterationError(f"unknown expert-iteration fixture kind {kind!r}")
    return dict(recipes[selected])


def expert_iteration_fixture_example(
    kind: str,
    *,
    example_id: str = "",
    split: str = "train",
) -> ExpertIterationExample:
    """Build one compact fixture example. Hidden-test kinds stay off the loop."""

    selected = _text(kind, "fixture kind").casefold()
    identity = example_id or f"example:{selected}"
    if selected == "hidden_test_feedback":
        # The example constructor rejects hidden splits.  Tests that need the
        # rejection path construct the raw mapping themselves.
        raise HiddenTestFeedbackError(
            "hidden-test fixture examples cannot be admitted onto the loop"
        )
    return ExpertIterationExample(
        example_id=identity,
        obligation_id=f"obligation:{identity}",
        split=split,
        statement=f"theorem {selected}",
        fixture_kind=selected,
        model_revision="model:candidate",
        tool_revision="tool:portfolio",
    )


# ---------------------------------------------------------------------------
# Stage ports
# ---------------------------------------------------------------------------


class ExpertIterationStageRunner(Protocol):
    def __call__(
        self,
        example: ExpertIterationExample,
        stage: ExpertIterationStage,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any] | None: ...


class ExpertIterationRefillPort(Protocol):
    def __call__(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        history: Mapping[str, Any],
        cursor_advanced: bool,
        progress_identity: str,
    ) -> Mapping[str, Any] | Any: ...


def default_stage_runner(
    example: ExpertIterationExample,
    stage: ExpertIterationStage,
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Deterministic default: fixture recipe, else an inconclusive candidate."""

    del stage, context
    if example.fixture_kind:
        return expert_iteration_fixture_payload(example.fixture_kind)
    return {"status": "failed", "reason_code": "inconclusive"}


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


class BoundedExpertIteration:
    """Round coordinator for the generate-check-retain-refill-train-qualify loop."""

    def __init__(
        self,
        *,
        campaign_id: str,
        bounds: ExpertIterationBounds | Mapping[str, Any] | None = None,
        state_path: Path | str | None = None,
        stage_runner: ExpertIterationStageRunner | None = None,
        refill_decide: ExpertIterationRefillPort | None = None,
        checkpoint_id: str = "",
    ) -> None:
        self.campaign_id = _text(campaign_id, "campaign_id")
        self.bounds = (
            bounds
            if isinstance(bounds, ExpertIterationBounds)
            else ExpertIterationBounds.from_dict(bounds)
        )
        path = Path(state_path) if state_path is not None else None
        if path is not None and path.suffix == "":
            path = path / DEFAULT_STATE_FILENAME
        self.state_path = path
        self.stage_runner = stage_runner or default_stage_runner
        self.refill_decide = refill_decide
        self.checkpoint_id = _optional_text(checkpoint_id, "checkpoint_id")
        self._checkpoint = ExpertIterationCheckpoint(
            campaign_id=self.campaign_id,
            bounds=self.bounds,
            checkpoint_id=self.checkpoint_id,
        )
        if self.state_path is not None and self.state_path.exists():
            self._checkpoint = ExpertIterationCheckpoint.load(self.state_path)
            if self._checkpoint.campaign_id != self.campaign_id:
                raise ExpertIterationError("checkpoint campaign_id does not match")
            if self._checkpoint.bounds.bounds_id != self.bounds.bounds_id:
                raise ExpertIterationError("checkpoint bounds do not match")

    @property
    def checkpoint(self) -> ExpertIterationCheckpoint:
        return self._checkpoint

    def run(
        self,
        examples: Sequence[ExpertIterationExample | Mapping[str, Any]] | None = None,
        *,
        halt_after_stage: ExpertIterationStage | str | None = None,
        halt_after_round: int | None = None,
    ) -> ExpertIterationResult:
        """Execute or resume the bounded loop until a terminal stop reason."""

        halt_stage = (
            _enum(halt_after_stage, ExpertIterationStage, "halt_after_stage")
            if halt_after_stage is not None
            else None
        )
        pending = self._admit_examples(examples)
        if pending:
            self._replace_pending(pending)
        stop = ExpertIterationStopReason.COMPLETED
        halted = False
        while True:
            if self._checkpoint.next_round_index >= self.bounds.max_rounds:
                stop = ExpertIterationStopReason.ROUND_BOUNDED
                break
            if self._checkpoint.no_progress_streak >= self.bounds.max_no_progress_rounds:
                stop = ExpertIterationStopReason.NO_PROGRESS
                break
            if not self._checkpoint.pending_examples:
                stop = ExpertIterationStopReason.COMPLETED
                break
            if self._checkpoint.call_count >= self.bounds.max_calls:
                stop = ExpertIterationStopReason.CALL_BOUNDED
                break
            if (
                halt_after_round is not None
                and self._checkpoint.next_round_index >= halt_after_round
            ):
                stop = ExpertIterationStopReason.HALTED
                halted = True
                break
            stop, halted = self._run_round(halt_stage=halt_stage)
            if halted or stop is not ExpertIterationStopReason.COMPLETED:
                break
        complete = stop is ExpertIterationStopReason.COMPLETED and not halted
        if complete and self._checkpoint.pending_examples:
            complete = False
        return self._result(stop_reason=stop, complete=complete)

    def resume(
        self,
        *,
        halt_after_stage: ExpertIterationStage | str | None = None,
        halt_after_round: int | None = None,
    ) -> ExpertIterationResult:
        return self.run(
            None,
            halt_after_stage=halt_after_stage,
            halt_after_round=halt_after_round,
        )

    def _admit_examples(
        self,
        examples: Sequence[ExpertIterationExample | Mapping[str, Any]] | None,
    ) -> tuple[ExpertIterationExample, ...]:
        if examples is None:
            return ()
        admitted: list[ExpertIterationExample] = []
        for raw in examples:
            if isinstance(raw, ExpertIterationExample):
                item = raw
            else:
                payload = _mapping(raw, "example")
                if _token(payload.get("split")) in HIDDEN_TEST_SPLITS or _token(
                    payload.get("fixture_kind")
                ) == "hidden_test_feedback":
                    raise HiddenTestFeedbackError(
                        "hidden-test examples cannot enter expert iteration"
                    )
                item = ExpertIterationExample.from_dict(payload)
            admitted.append(item)
        if len(admitted) > self.bounds.max_candidates:
            raise ExpertIterationBoundError("candidate bound exceeded")
        return tuple(admitted)

    def _replace_pending(self, examples: Sequence[ExpertIterationExample]) -> None:
        current = ExpertIterationCheckpoint.from_dict(self._checkpoint.to_dict())
        object.__setattr__(
            current,
            "pending_examples",
            tuple(item.to_dict() for item in examples),
        )
        self._persist(current, sealed_stage="")

    def _run_round(
        self,
        *,
        halt_stage: ExpertIterationStage | None,
    ) -> tuple[ExpertIterationStopReason, bool]:
        examples = tuple(
            ExpertIterationExample.from_dict(item)
            for item in self._checkpoint.pending_examples
        )
        if len(examples) > self.bounds.max_candidates:
            self._seal_bound(ExpertIterationStopReason.CANDIDATE_BOUNDED)
            return ExpertIterationStopReason.CANDIDATE_BOUNDED, False

        attempts: list[ExpertIterationAttempt] = []
        retained: list[ExpertIterationAttempt] = []
        residuals: list[ExpertIterationExample] = []
        for example in examples:
            if example.example_id in self._checkpoint.completed_example_ids:
                continue
            seen = int(self._checkpoint.curriculum_repetitions.get(example.example_id, 0))
            if seen >= self.bounds.max_repeated_examples:
                self._seal_bound(ExpertIterationStopReason.REPETITION_BOUNDED)
                return ExpertIterationStopReason.REPETITION_BOUNDED, False
            attempt, halt = self._run_example_pipeline(example, halt_stage=halt_stage)
            if halt:
                return ExpertIterationStopReason.HALTED, True
            if attempt is None:
                continue
            attempts.append(attempt)
            if attempt.retained:
                retained.append(attempt)
            elif attempt.outcome_class is not ExpertIterationOutcomeClass.HIDDEN_TEST_FEEDBACK:
                residuals.append(example)
            self._record_example(example, attempt)
            if self._checkpoint.call_count >= self.bounds.max_calls:
                self._seal_bound(ExpertIterationStopReason.CALL_BOUNDED)
                return ExpertIterationStopReason.CALL_BOUNDED, False
            if self._checkpoint.solver_time_ms >= self.bounds.max_solver_time_ms:
                self._seal_bound(ExpertIterationStopReason.SOLVER_TIME_BOUNDED)
                return ExpertIterationStopReason.SOLVER_TIME_BOUNDED, False

        refill_candidates, refill_disposition, next_examples = self._refill_stage(
            residuals
        )
        if halt_stage is ExpertIterationStage.REFILL:
            self._persist(self._checkpoint, sealed_stage=ExpertIterationStage.REFILL.value)
            return ExpertIterationStopReason.HALTED, True

        progress = bool(retained)
        no_progress = self._checkpoint.no_progress_streak + (0 if progress else 1)
        if not progress:
            no_progress = self._checkpoint.no_progress_streak + 1
        else:
            no_progress = 0

        revision = CurriculumRevision(
            campaign_id=self.campaign_id,
            round_index=self._checkpoint.next_round_index,
            retained_attempt_ids=tuple(item.attempt_id for item in retained),
            residual_example_ids=tuple(item.example_id for item in residuals),
            parent_revision_id=self._checkpoint.parent_revision_id,
            checkpoint_id=self.checkpoint_id,
        )
        train_payload = self._invoke(
            residuals[0] if residuals else examples[0],
            ExpertIterationStage.TRAIN,
            {"retained": [item.to_dict() for item in retained]},
        )
        _reject_promotion_authority(train_payload, noun="train stage")
        if halt_stage is ExpertIterationStage.TRAIN:
            self._persist(self._checkpoint, sealed_stage=ExpertIterationStage.TRAIN.value)
            return ExpertIterationStopReason.HALTED, True
        qualify_payload = self._invoke(
            residuals[0] if residuals else examples[0],
            ExpertIterationStage.QUALIFY,
            {"curriculum_revision_id": revision.revision_id},
        )
        _reject_promotion_authority(qualify_payload, noun="qualify stage")
        if _truthy(qualify_payload.get("promote")) or _truthy(
            qualify_payload.get("self_promote")
        ):
            raise CheckpointSelfPromotionError("qualify cannot self-promote a checkpoint")

        stop = ExpertIterationStopReason.COMPLETED
        if no_progress >= self.bounds.max_no_progress_rounds:
            stop = ExpertIterationStopReason.NO_PROGRESS
        elif refill_disposition in {"no_progress_bounded", "round_bounded", "repetition_bounded"}:
            stop = {
                "no_progress_bounded": ExpertIterationStopReason.NO_PROGRESS,
                "round_bounded": ExpertIterationStopReason.ROUND_BOUNDED,
                "repetition_bounded": ExpertIterationStopReason.REPETITION_BOUNDED,
            }[refill_disposition]
        elif not next_examples and not retained:
            stop = ExpertIterationStopReason.EXHAUSTED

        receipt = ExpertIterationRoundReceipt(
            campaign_id=self.campaign_id,
            round_index=self._checkpoint.next_round_index,
            bounds_id=self.bounds.bounds_id,
            curriculum_revision_id=revision.revision_id,
            outcome_classes=tuple(item.outcome_class.value for item in attempts),
            retained_count=len(retained),
            residual_count=len(residuals),
            call_count=self._checkpoint.call_count,
            solver_time_ms=self._checkpoint.solver_time_ms,
            no_progress_streak=no_progress,
            stop_reason=stop,
            refill_disposition=refill_disposition,
            attempt_ids=tuple(item.attempt_id for item in attempts),
        )
        updated = ExpertIterationCheckpoint.from_dict(self._checkpoint.to_dict())
        object.__setattr__(updated, "next_round_index", updated.next_round_index + 1)
        object.__setattr__(updated, "sealed_stage", ExpertIterationStage.QUALIFY.value)
        object.__setattr__(updated, "no_progress_streak", no_progress)
        object.__setattr__(
            updated,
            "last_progress_identity",
            retained[-1].example_id if retained else updated.last_progress_identity,
        )
        object.__setattr__(
            updated,
            "pending_examples",
            tuple(item.to_dict() for item in next_examples),
        )
        object.__setattr__(updated, "completed_example_ids", ())
        object.__setattr__(
            updated,
            "retained_attempt_ids",
            tuple(dict.fromkeys((*updated.retained_attempt_ids, *(item.attempt_id for item in retained)))),
        )
        object.__setattr__(
            updated,
            "residual_example_ids",
            tuple(item.example_id for item in residuals),
        )
        object.__setattr__(
            updated,
            "receipt_payloads",
            (*updated.receipt_payloads, receipt.to_dict()),
        )
        object.__setattr__(
            updated,
            "curriculum_revision_ids",
            (*updated.curriculum_revision_ids, revision.revision_id),
        )
        object.__setattr__(updated, "parent_revision_id", revision.revision_id)
        object.__setattr__(
            updated,
            "revision_payloads",
            (*updated.revision_payloads, revision.to_dict()),
        )
        object.__setattr__(
            updated,
            "refill_payloads",
            tuple(item.to_dict() for item in refill_candidates),
        )
        self._persist(updated, sealed_stage=ExpertIterationStage.QUALIFY.value)
        if halt_stage is ExpertIterationStage.QUALIFY:
            return ExpertIterationStopReason.HALTED, True
        return stop, False

    def _run_example_pipeline(
        self,
        example: ExpertIterationExample,
        *,
        halt_stage: ExpertIterationStage | None,
    ) -> tuple[ExpertIterationAttempt | None, bool]:
        context: dict[str, Any] = {
            "campaign_id": self.campaign_id,
            "round_index": self._checkpoint.next_round_index,
            "depth": 1,
        }
        last: ExpertIterationAttempt | None = None
        for stage in (
            ExpertIterationStage.GENERATE,
            ExpertIterationStage.PARSE_TYPE,
            ExpertIterationStage.TACTICIAN,
            ExpertIterationStage.HAMMER,
            ExpertIterationStage.CHECK,
            ExpertIterationStage.RETAIN,
        ):
            if stage is ExpertIterationStage.TACTICIAN and context["depth"] > self.bounds.max_depth:
                self._seal_bound(ExpertIterationStopReason.DEPTH_BOUNDED)
                raise ExpertIterationBoundError("depth bound exceeded")
            payload = self._invoke(example, stage, context)
            solver_ms = int(payload.get("solver_time_ms") or 0)
            if solver_ms:
                self._add_solver_time(solver_ms)
            if stage is not ExpertIterationStage.RETAIN:
                self._add_call(1)
            last = build_attempt(
                example,
                payload,
                stage=stage,
                depth=int(context["depth"]),
                call_count=self._checkpoint.call_count,
                solver_time_ms=self._checkpoint.solver_time_ms,
            )
            context["last_attempt"] = last.to_dict()
            if stage is ExpertIterationStage.TACTICIAN:
                context["depth"] = int(payload.get("depth") or context["depth"])
                if context["depth"] > self.bounds.max_depth:
                    raise ExpertIterationBoundError("depth bound exceeded")
            if halt_stage is stage:
                self._persist(self._checkpoint, sealed_stage=stage.value)
                return last, True
            if last.outcome_class in {
                ExpertIterationOutcomeClass.TIMEOUT,
                ExpertIterationOutcomeClass.UNAVAILABLE,
                ExpertIterationOutcomeClass.UNSUPPORTED,
                ExpertIterationOutcomeClass.PARSE_TYPE_FAILURE,
                ExpertIterationOutcomeClass.HIDDEN_TEST_FEEDBACK,
            } and stage is not ExpertIterationStage.RETAIN:
                last = build_attempt(
                    example,
                    payload,
                    stage=ExpertIterationStage.RETAIN,
                    depth=int(context["depth"]),
                    call_count=self._checkpoint.call_count,
                    solver_time_ms=self._checkpoint.solver_time_ms,
                )
                break
        return last, False

    def _invoke(
        self,
        example: ExpertIterationExample,
        stage: ExpertIterationStage,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw = self.stage_runner(example, stage, context)
        if raw is None:
            if example.fixture_kind:
                return expert_iteration_fixture_payload(example.fixture_kind)
            return {"status": "failed", "reason_code": "inconclusive"}
        payload = _mapping(raw, f"{stage.value} stage")
        _reject_hidden_feedback(payload, noun=f"{stage.value} stage")
        _reject_promotion_authority(payload, noun=f"{stage.value} stage")
        return payload

    def _refill_stage(
        self,
        residuals: Sequence[ExpertIterationExample],
    ) -> tuple[
        tuple[ExpertIterationRefillCandidate, ...],
        str,
        tuple[ExpertIterationExample, ...],
    ]:
        candidates = tuple(
            ExpertIterationRefillCandidate(
                candidate_id=item.example_id,
                trigger="proof_residual",
                residual_count=1,
                curriculum_key=item.example_id,
                metadata={"obligation_id": item.obligation_id},
            )
            for item in residuals
        )
        history = {
            "refill_rounds": self._checkpoint.next_round_index,
            "open_work": len(residuals),
            "no_progress_streak": self._checkpoint.no_progress_streak,
            "last_progress_identity": self._checkpoint.last_progress_identity,
            "curriculum_repetitions": dict(self._checkpoint.curriculum_repetitions),
        }
        progress_identity = residuals[0].example_id if residuals else ""
        cursor_advanced = bool(self._checkpoint.retained_attempt_ids) and bool(
            residuals
        ) is False
        # Cursor advances only when the just-finished round retained work.
        cursor_advanced = False
        if self.refill_decide is None:
            if not candidates:
                return (), "no_admissible_candidates", ()
            if self._checkpoint.next_round_index + 1 >= self.bounds.max_rounds:
                return candidates, "round_bounded", ()
            if (
                self._checkpoint.no_progress_streak + 1
                >= self.bounds.max_no_progress_rounds
            ):
                return candidates, "no_progress_bounded", ()
            admitted: list[ExpertIterationExample] = []
            for item in residuals:
                seen = int(self._checkpoint.curriculum_repetitions.get(item.example_id, 0))
                if seen >= self.bounds.max_repeated_examples:
                    return candidates, "repetition_bounded", ()
                admitted.append(item)
            return candidates, "admitted", tuple(admitted)

        decision = self.refill_decide(
            [item.to_dict() for item in candidates],
            history=history,
            cursor_advanced=cursor_advanced,
            progress_identity=progress_identity,
        )
        payload = (
            decision.to_dict()
            if hasattr(decision, "to_dict")
            else _mapping(decision, "refill decision")
        )
        disposition = str(
            getattr(getattr(decision, "disposition", None), "value", None)
            or payload.get("disposition")
            or payload.get("reason_code")
            or ""
        ).casefold()
        admitted_raw = getattr(decision, "admitted", None)
        if admitted_raw is None:
            admitted_raw = payload.get("admitted") or ()
        by_id = {item.example_id: item for item in residuals}
        next_examples: list[ExpertIterationExample] = []
        for raw in admitted_raw:
            candidate_id = (
                raw.candidate_id
                if hasattr(raw, "candidate_id")
                else str(_mapping(raw, "admitted").get("candidate_id") or "")
            )
            example = by_id.get(candidate_id)
            if example is None:
                continue
            seen = int(self._checkpoint.curriculum_repetitions.get(example.example_id, 0))
            if seen >= self.bounds.max_repeated_examples:
                return candidates, "repetition_bounded", ()
            next_examples.append(example)
        return candidates, disposition or "admitted", tuple(next_examples)

    def _record_example(
        self,
        example: ExpertIterationExample,
        attempt: ExpertIterationAttempt,
    ) -> None:
        updated = ExpertIterationCheckpoint.from_dict(self._checkpoint.to_dict())
        object.__setattr__(
            updated,
            "completed_example_ids",
            tuple(dict.fromkeys((*updated.completed_example_ids, example.example_id))),
        )
        object.__setattr__(
            updated,
            "attempt_payloads",
            (*updated.attempt_payloads, attempt.to_dict()),
        )
        repetitions = dict(updated.curriculum_repetitions)
        repetitions[example.example_id] = int(repetitions.get(example.example_id, 0)) + 1
        object.__setattr__(updated, "curriculum_repetitions", repetitions)
        self._persist(updated, sealed_stage=ExpertIterationStage.RETAIN.value)

    def _add_call(self, count: int) -> None:
        updated = ExpertIterationCheckpoint.from_dict(self._checkpoint.to_dict())
        object.__setattr__(updated, "call_count", updated.call_count + count)
        self._checkpoint = updated

    def _add_solver_time(self, millis: int) -> None:
        updated = ExpertIterationCheckpoint.from_dict(self._checkpoint.to_dict())
        object.__setattr__(updated, "solver_time_ms", updated.solver_time_ms + millis)
        self._checkpoint = updated

    def _seal_bound(self, reason: ExpertIterationStopReason) -> None:
        updated = ExpertIterationCheckpoint.from_dict(self._checkpoint.to_dict())
        object.__setattr__(updated, "sealed_stage", reason.value)
        self._persist(updated, sealed_stage=reason.value)

    def _persist(
        self,
        checkpoint: ExpertIterationCheckpoint,
        *,
        sealed_stage: str,
    ) -> None:
        object.__setattr__(checkpoint, "fence", checkpoint.fence + 1)
        object.__setattr__(checkpoint, "sealed_stage", sealed_stage)
        object.__setattr__(checkpoint, "promotion_authority", False)
        if self.state_path is not None:
            checkpoint.write(self.state_path)
        self._checkpoint = checkpoint

    def _result(
        self,
        *,
        stop_reason: ExpertIterationStopReason,
        complete: bool,
    ) -> ExpertIterationResult:
        attempts = tuple(
            ExpertIterationAttempt.from_dict(item)
            for item in self._checkpoint.attempt_payloads
        )
        receipts = tuple(
            ExpertIterationRoundReceipt.from_dict(item)
            for item in self._checkpoint.receipt_payloads
        )
        revisions = tuple(
            CurriculumRevision.from_dict(item)
            for item in self._checkpoint.revision_payloads
        )
        refill_payloads = self._checkpoint.refill_payloads
        refill = tuple(
            ExpertIterationRefillCandidate(
                candidate_id=str(item.get("candidate_id") or ""),
                trigger=str(item.get("trigger") or "proof_residual"),
                residual_count=int(item.get("residual_count") or 1),
                curriculum_key=str(item.get("curriculum_key") or ""),
                metadata=item.get("metadata") or {},
            )
            for item in refill_payloads
        )
        return ExpertIterationResult(
            campaign_id=self.campaign_id,
            complete=complete,
            stop_reason=stop_reason,
            bounds=self.bounds,
            attempts=attempts,
            receipts=receipts,
            curriculum_revisions=revisions,
            refill_candidates=refill,
            no_progress_streak=self._checkpoint.no_progress_streak,
            call_count=self._checkpoint.call_count,
            solver_time_ms=self._checkpoint.solver_time_ms,
            fence=self._checkpoint.fence,
        )


def run_bounded_expert_iteration(
    examples: Sequence[ExpertIterationExample | Mapping[str, Any]],
    *,
    campaign_id: str,
    bounds: ExpertIterationBounds | Mapping[str, Any] | None = None,
    state_path: Path | str | None = None,
    stage_runner: ExpertIterationStageRunner | None = None,
    refill_decide: ExpertIterationRefillPort | None = None,
    checkpoint_id: str = "",
    halt_after_stage: ExpertIterationStage | str | None = None,
    halt_after_round: int | None = None,
) -> ExpertIterationResult:
    """Convenience entry point for one bounded expert-iteration run."""

    loop = BoundedExpertIteration(
        campaign_id=campaign_id,
        bounds=bounds,
        state_path=state_path,
        stage_runner=stage_runner,
        refill_decide=refill_decide,
        checkpoint_id=checkpoint_id,
    )
    return loop.run(
        examples,
        halt_after_stage=halt_after_stage,
        halt_after_round=halt_after_round,
    )


__all__ = (
    "ADMITTED_SPLITS",
    "BOUNDED_EXPERT_ITERATION_INTERFACE",
    "BOUNDED_EXPERT_ITERATION_SCHEMA",
    "CURRICULUM_REVISION_SCHEMA",
    "EXPERT_ITERATION_ATTEMPT_SCHEMA",
    "EXPERT_ITERATION_BOUNDS_SCHEMA",
    "EXPERT_ITERATION_CHECKPOINT_SCHEMA",
    "EXPERT_ITERATION_FIXTURE_KINDS",
    "EXPERT_ITERATION_RESULT_SCHEMA",
    "EXPERT_ITERATION_ROUND_RECEIPT_SCHEMA",
    "HIDDEN_TEST_SPLITS",
    "MAX_CANDIDATES",
    "MAX_CALLS",
    "MAX_DEPTH",
    "MAX_NO_PROGRESS_ROUNDS",
    "MAX_REPEATED_EXAMPLES",
    "MAX_ROUNDS",
    "MAX_SOLVER_TIME_MS",
    "BoundedExpertIteration",
    "CheckpointSelfPromotionError",
    "CurriculumRevision",
    "ExpertIterationAttempt",
    "ExpertIterationBoundError",
    "ExpertIterationBounds",
    "ExpertIterationCheckpoint",
    "ExpertIterationError",
    "ExpertIterationExample",
    "ExpertIterationOutcomeClass",
    "ExpertIterationRefillCandidate",
    "ExpertIterationResult",
    "ExpertIterationRoundReceipt",
    "ExpertIterationStage",
    "ExpertIterationStopReason",
    "HiddenTestFeedbackError",
    "UnverifiedRetentionError",
    "build_attempt",
    "curriculum_authority_for",
    "expert_iteration_fixture_example",
    "expert_iteration_fixture_payload",
    "map_attempt_outcome",
    "run_bounded_expert_iteration",
)
