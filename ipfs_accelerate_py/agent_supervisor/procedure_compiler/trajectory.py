"""Trajectory admission, redaction, and P0 contract validation.

P0 ships contract validation for already-constructed, independently admitted
trajectory contracts.  PCPC-009 adds :class:`TrajectoryNormalizer` and
:class:`TrajectoryAdmissionPolicy` in this same module; the contract helpers
remain the only decoder for already-normalized wire artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .contracts import (
    FORBIDDEN_HOLE_TYPES,
    FORBIDDEN_STEP_OPERATIONS,
    MAX_STEPS,
    MAX_STRUCTURED_INTEGER,
    ArtifactBindings,
    EpisodeKind,
    ExecutionTrajectory,
    HoleType,
    ProcedureContractError,
    StepOperation,
    TraceEventStatus,
    TrajectoryNormalizationReceipt,
    TrajectoryOutcome,
    TrajectoryStep,
    TrajectoryTerminalStatus,
    _enum,
    _identifier,
    _nested,
    _nonnegative_int,
    _strings,
)

TRAJECTORY_NORMALIZER_REVISION: Final[str] = "TrajectoryNormalizer@1"
SOURCE_EPISODE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/procedure-compiler/source-episode@1"
)
TRAJECTORY_ADMISSION_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/procedure-compiler/trajectory-admission-policy@1"
)
DEFAULT_MAX_RECEIPT_AGE_MS: Final[int] = 86_400_000

ADMISSIBLE_SOURCE_EPISODE_KINDS: Final[frozenset[EpisodeKind]] = frozenset(EpisodeKind)
SUCCESS_DEMONSTRATION_SOURCE_KINDS: Final[frozenset[EpisodeKind]] = frozenset(
    {
        EpisodeKind.ACCEPTED_TASK_RECEIPT,
        EpisodeKind.CURRENT_TREE_POST_MERGE_RECEIPT,
        EpisodeKind.VERIFIED_PROOF_RECEIPT,
        EpisodeKind.ADMITTED_TEST_RECEIPT,
        EpisodeKind.SUCCESSFUL_ROLLBACK_RECEIPT,
        EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT,
    }
)

LIVE_PRODUCTION_MODES: Final[frozenset[str]] = frozenset({"live", "production"})
DEFAULT_REJECTION_REASON: Final[str] = "typed_rejection"


class RemovedFieldClass(str, Enum):
    """Closed classes of private or unbounded source material that must be stripped."""

    PROMPT = "prompt"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SECRET = "secret"
    CREDENTIAL = "credential"
    BODY = "body"
    LOG = "log"


class TrajectoryAdmissionReason(str, Enum):
    """Closed, machine-actionable admission failures."""

    MALFORMED_EPISODE = "malformed_episode"
    INADMISSIBLE_SOURCE_KIND = "inadmissible_source_kind"
    PROSE_EVIDENCE = "prose_evidence"
    BOARD_STATUS_EVIDENCE = "board_status_evidence"
    MODEL_CONFIDENCE_EVIDENCE = "model_confidence_evidence"
    UNSIGNED_RECEIPT = "unsigned_receipt"
    STALE_RECEIPT = "stale_receipt"
    SIMULATED_PRODUCTION = "simulated_production"
    PRE_MERGE_ONLY_VALIDATION = "pre_merge_only_validation"
    MISSING_VALIDATION = "missing_validation"
    MISSING_STEPS = "missing_steps"
    MISSING_OBSERVATION = "missing_observation"
    MISSING_EFFECT = "missing_effect"
    FORBIDDEN_OPERATION = "forbidden_operation"
    FORBIDDEN_HOLE = "forbidden_hole"
    KIND_SHAPE_MISMATCH = "kind_shape_mismatch"
    INCONSISTENT_COST = "inconsistent_cost"
    UNSUPPORTED_FIELD = "unsupported_field"


class TrajectoryContractError(ProcedureContractError):
    """An already-normalized trajectory violates the P0 wire contract."""


class TrajectoryAdmissionError(TrajectoryContractError):
    """A source episode cannot be admitted for trajectory normalization."""

    def __init__(self, message: str, *, reason_code: str = "") -> None:
        super().__init__(message)
        self.reason_code = reason_code or _reason_from_message(message)


def _reason_from_message(message: str) -> str:
    for reason in TrajectoryAdmissionReason:
        if reason.value.replace("_", " ") in message or reason.value in message:
            return reason.value
    return TrajectoryAdmissionReason.MALFORMED_EPISODE.value


_FORBIDDEN_EVIDENCE_FIELDS: Final[dict[str, TrajectoryAdmissionReason]] = {
    "prose": TrajectoryAdmissionReason.PROSE_EVIDENCE,
    "narrative": TrajectoryAdmissionReason.PROSE_EVIDENCE,
    "board_status": TrajectoryAdmissionReason.BOARD_STATUS_EVIDENCE,
    "board": TrajectoryAdmissionReason.BOARD_STATUS_EVIDENCE,
    "model_confidence": TrajectoryAdmissionReason.MODEL_CONFIDENCE_EVIDENCE,
    "confidence_score": TrajectoryAdmissionReason.MODEL_CONFIDENCE_EVIDENCE,
}

_FORBIDDEN_SOURCE_CLASSES: Final[dict[str, TrajectoryAdmissionReason]] = {
    "prose": TrajectoryAdmissionReason.PROSE_EVIDENCE,
    "narrative": TrajectoryAdmissionReason.PROSE_EVIDENCE,
    "board_status": TrajectoryAdmissionReason.BOARD_STATUS_EVIDENCE,
    "board": TrajectoryAdmissionReason.BOARD_STATUS_EVIDENCE,
    "model_confidence": TrajectoryAdmissionReason.MODEL_CONFIDENCE_EVIDENCE,
    "confidence": TrajectoryAdmissionReason.MODEL_CONFIDENCE_EVIDENCE,
    "simulated": TrajectoryAdmissionReason.SIMULATED_PRODUCTION,
    "pre_merge_only": TrajectoryAdmissionReason.PRE_MERGE_ONLY_VALIDATION,
    "unsigned": TrajectoryAdmissionReason.UNSIGNED_RECEIPT,
    "stale": TrajectoryAdmissionReason.STALE_RECEIPT,
}

_EXACT_REDACTION_KEYS: Final[dict[str, RemovedFieldClass]] = {
    "prompt": RemovedFieldClass.PROMPT,
    "private_prompt": RemovedFieldClass.PROMPT,
    "model_prompt": RemovedFieldClass.PROMPT,
    "system_prompt": RemovedFieldClass.PROMPT,
    "messages": RemovedFieldClass.PROMPT,
    "model_messages": RemovedFieldClass.PROMPT,
    "chain_of_thought": RemovedFieldClass.CHAIN_OF_THOUGHT,
    "chainofthought": RemovedFieldClass.CHAIN_OF_THOUGHT,
    "reasoning_trace": RemovedFieldClass.CHAIN_OF_THOUGHT,
    "private_reasoning": RemovedFieldClass.CHAIN_OF_THOUGHT,
    "secret": RemovedFieldClass.SECRET,
    "secrets": RemovedFieldClass.SECRET,
    "api_key": RemovedFieldClass.SECRET,
    "authorization": RemovedFieldClass.SECRET,
    "cookie": RemovedFieldClass.SECRET,
    "password": RemovedFieldClass.SECRET,
    "private_key": RemovedFieldClass.SECRET,
    "refresh_token": RemovedFieldClass.SECRET,
    "session_token": RemovedFieldClass.SECRET,
    "credential": RemovedFieldClass.CREDENTIAL,
    "credentials": RemovedFieldClass.CREDENTIAL,
    "body": RemovedFieldClass.BODY,
    "source_body": RemovedFieldClass.BODY,
    "redundant_body": RemovedFieldClass.BODY,
    "code_body": RemovedFieldClass.BODY,
    "source_code": RemovedFieldClass.BODY,
    "log": RemovedFieldClass.LOG,
    "logs": RemovedFieldClass.LOG,
    "unbounded_log": RemovedFieldClass.LOG,
    "unbounded_logs": RemovedFieldClass.LOG,
    "transcript": RemovedFieldClass.LOG,
    "model_transcript": RemovedFieldClass.LOG,
}

_SECRET_SUBSTRINGS: Final[frozenset[str]] = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "session_token",
    }
)

_EPISODE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "contract_version",
        "content_id",
        "cid",
        "episode_cid",
        "source_episode_cid",
        "episode_kind",
        "source_episode_kind",
        "bindings",
        "signature",
        "signed",
        "unsigned",
        "current",
        "simulated",
        "stale",
        "pre_merge_only",
        "issued_at_ms",
        "expires_at_ms",
        "initial_abstract_state_cid",
        "terminal_abstract_state_cid",
        "objective_criterion_ids",
        "task_family_hint",
        "accepted_criterion_ids",
        "validation_receipt_cids",
        "proof_receipt_cids",
        "rejection_reason_code",
        "outcome_status",
        "steps",
        "total_cost_units",
        "total_tokens",
        "total_latency_ms",
        "human_interventions",
        "admitted_evidence_cids",
        "source_class",
        "evidence_class",
        "production_mode",
    }
)

_STEP_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "contract_version",
        "content_id",
        "cid",
        "sequence",
        "operation",
        "operation_contract",
        "initial_state_cid",
        "terminal_state_cid",
        "observation_cids",
        "effect_ids",
        "validation_receipt_cids",
        "hole_type",
        "model_calls",
        "input_tokens",
        "output_tokens",
        "latency_ms",
        "human_interventions",
        "status",
        "cost_units",
    }
)

_KIND_REQUIRED_ANY_OPERATION: Final[dict[EpisodeKind, frozenset[StepOperation]]] = {
    EpisodeKind.CURRENT_TREE_POST_MERGE_RECEIPT: frozenset(
        {
            StepOperation.PREPARE_MERGE,
            StepOperation.MERGE_IN_ISOLATED_TRAIN,
            StepOperation.VERIFY_MERGED_TREE,
        }
    ),
    EpisodeKind.VERIFIED_PROOF_RECEIPT: frozenset({StepOperation.RUN_PROOF}),
    EpisodeKind.ADMITTED_TEST_RECEIPT: frozenset(
        {StepOperation.RUN_SELECTED_TESTS, StepOperation.RUN_FULL_TEST_FALLBACK}
    ),
    EpisodeKind.SUCCESSFUL_ROLLBACK_RECEIPT: frozenset({StepOperation.ROLLBACK}),
    EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT: frozenset(
        {StepOperation.ESCALATE, StepOperation.CHECK_AUTHORITY}
    ),
}

_DEFAULT_OUTCOME_STATUS: Final[dict[EpisodeKind, TrajectoryTerminalStatus]] = {
    EpisodeKind.REJECTED_TASK_RECORD: TrajectoryTerminalStatus.REJECTED,
    EpisodeKind.FAILED_RECOVERED_EXECUTION: TrajectoryTerminalStatus.FAILED_RECOVERED,
    EpisodeKind.SUCCESSFUL_ROLLBACK_RECEIPT: TrajectoryTerminalStatus.ROLLED_BACK,
}


def validate_execution_trajectory_contract(
    trajectory: ExecutionTrajectory,
) -> ExecutionTrajectory:
    """Validate chain, cost, and admitted-outcome consistency.

    This is deliberately not an admission function: it accepts only the typed
    immutable contract and never upgrades candidate evidence.
    """

    if not isinstance(trajectory, ExecutionTrajectory):
        raise TrajectoryContractError("trajectory must be ExecutionTrajectory")
    if trajectory.source_episode_kind not in ADMISSIBLE_SOURCE_EPISODE_KINDS:
        raise TrajectoryContractError("trajectory source kind is not admissible")
    if trajectory.steps[0].initial_state_cid != trajectory.initial_abstract_state_cid:
        raise TrajectoryContractError("first step does not bind the declared initial state")
    if trajectory.steps[-1].terminal_state_cid != trajectory.terminal_abstract_state_cid:
        raise TrajectoryContractError("last step does not bind the declared terminal state")
    for previous, current in zip(trajectory.steps, trajectory.steps[1:], strict=False):
        if previous.terminal_state_cid != current.initial_state_cid:
            raise TrajectoryContractError("trajectory state chain is discontinuous")

    step_tokens = sum(step.input_tokens + step.output_tokens for step in trajectory.steps)
    step_latency = sum(step.latency_ms for step in trajectory.steps)
    step_humans = sum(step.human_interventions for step in trajectory.steps)
    if trajectory.total_tokens != step_tokens:
        raise TrajectoryContractError("trajectory token total is not denominator-preserving")
    if trajectory.total_latency_ms < step_latency:
        raise TrajectoryContractError("trajectory latency omits step latency")
    if trajectory.human_interventions != step_humans:
        raise TrajectoryContractError("trajectory human-intervention total is inconsistent")
    for step in trajectory.steps:
        if step.model_calls == 0 and (step.input_tokens or step.output_tokens):
            raise TrajectoryContractError("tokens cannot be attributed without a model call")
        if step.model_calls and not step.hole_type:
            raise TrajectoryContractError("model calls must be attributed to a typed hole")
        if step.hole_type:
            try:
                HoleType(step.hole_type)
            except ValueError as exc:
                raise TrajectoryContractError("trajectory names an unknown hole type") from exc

    outcome = trajectory.outcome
    if outcome.status is TrajectoryTerminalStatus.ACCEPTED:
        if trajectory.source_episode_kind not in SUCCESS_DEMONSTRATION_SOURCE_KINDS:
            raise TrajectoryContractError("source kind cannot demonstrate accepted success")
        if not set(outcome.accepted_criterion_ids).issubset(
            set(trajectory.objective_criterion_ids)
        ):
            raise TrajectoryContractError(
                "outcome claims criteria outside the exact objective subset"
            )
        step_validation = {
            receipt for step in trajectory.steps for receipt in step.validation_receipt_cids
        }
        if not step_validation.issubset(set(outcome.validation_receipt_cids)):
            raise TrajectoryContractError("accepted outcome omits step validation evidence")
    return trajectory


def _closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrajectoryContractError("trajectory JSON contains a duplicate field")
        result[key] = value
    return result


def _reject_float(_: str) -> Any:
    raise TrajectoryContractError("trajectory JSON cannot contain floating point values")


def parse_execution_trajectory(value: Any) -> ExecutionTrajectory:
    """Decode the closed trajectory schema and run contract-only checks."""

    if isinstance(value, ExecutionTrajectory):
        return validate_execution_trajectory_contract(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise TrajectoryContractError("trajectory bytes must be UTF-8") from exc
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                object_pairs_hook=_closed_object,
                parse_float=_reject_float,
                parse_constant=_reject_float,
            )
        except json.JSONDecodeError as exc:
            raise TrajectoryContractError("trajectory JSON is malformed") from exc
    if not isinstance(value, Mapping):
        raise TrajectoryContractError("trajectory must be a mapping or JSON object")
    return validate_execution_trajectory_contract(ExecutionTrajectory.from_dict(value))


def _admission_error(reason: TrajectoryAdmissionReason, message: str) -> TrajectoryAdmissionError:
    return TrajectoryAdmissionError(message, reason_code=reason.value)


def _decode_source_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, ExecutionTrajectory):
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            "an already-normalized trajectory is not a source episode",
        )
    if isinstance(value, TrajectoryNormalizationReceipt):
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            "a normalization receipt is not a source episode",
        )
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise _admission_error(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                "source episode bytes must be UTF-8",
            ) from exc
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                object_pairs_hook=_closed_object,
                parse_float=_reject_float,
                parse_constant=_reject_float,
            )
        except json.JSONDecodeError as exc:
            raise _admission_error(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                "source episode JSON is malformed",
            ) from exc
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict) and not isinstance(value, Mapping):
        value = to_dict()
    if not isinstance(value, Mapping):
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            "source episode must be a mapping or JSON object",
        )
    decoded: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise _admission_error(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                "source episode keys must be strings",
            )
        decoded[key] = item
    schema = decoded.get("schema")
    if schema not in (None, "", SOURCE_EPISODE_SCHEMA):
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            "source episode has an unsupported schema",
        )
    return decoded


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def _redaction_class(key: str) -> RemovedFieldClass | None:
    normalized = _normalize_key(key)
    if normalized in _EXACT_REDACTION_KEYS:
        return _EXACT_REDACTION_KEYS[normalized]
    compact = normalized.replace("_", "")
    for exact, field_class in _EXACT_REDACTION_KEYS.items():
        if exact.replace("_", "") == compact:
            return field_class
    for marker in _SECRET_SUBSTRINGS:
        if marker in normalized:
            return (
                RemovedFieldClass.CREDENTIAL
                if "credential" in marker
                else RemovedFieldClass.SECRET
            )
    return None


def _redact_mapping(
    payload: Mapping[str, Any],
    *,
    allowed: frozenset[str] | None,
    path: str,
) -> tuple[dict[str, Any], list[str]]:
    removed: list[str] = []
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        field_class = _redaction_class(key)
        if field_class is not None:
            removed.append(field_class.value)
            continue
        forbidden = _FORBIDDEN_EVIDENCE_FIELDS.get(_normalize_key(key))
        if forbidden is not None:
            raise _admission_error(
                forbidden,
                f"{path} offers {forbidden.value.replace('_', ' ')}",
            )
        if allowed is not None and key not in allowed:
            raise _admission_error(
                TrajectoryAdmissionReason.UNSUPPORTED_FIELD,
                f"{path} contains unsupported field {key}",
            )
        if isinstance(value, Mapping):
            if key == "bindings":
                nested_allowed: frozenset[str] | None = None
            elif key == "steps":
                nested_allowed = _STEP_FIELDS
            else:
                nested_allowed = allowed
            nested, nested_removed = _redact_mapping(
                value,
                allowed=nested_allowed,
                path=f"{path}.{key}",
            )
            cleaned[key] = nested
            removed.extend(nested_removed)
            continue
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray, memoryview)
        ):
            items: list[Any] = []
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    nested, nested_removed = _redact_mapping(
                        item,
                        allowed=_STEP_FIELDS if key == "steps" else allowed,
                        path=f"{path}.{key}[{index}]",
                    )
                    items.append(nested)
                    removed.extend(nested_removed)
                else:
                    items.append(item)
            cleaned[key] = items
            continue
        cleaned[key] = value
    return cleaned, removed


def _flag(payload: Mapping[str, Any], name: str, *, default: bool = False) -> bool:
    if name not in payload:
        return default
    value = payload[name]
    if type(value) is not bool:
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            f"{name} must be a boolean",
        )
    return value


def _optional_int(payload: Mapping[str, Any], name: str) -> int | None:
    if name not in payload or payload[name] is None:
        return None
    try:
        return _nonnegative_int(payload[name], name, maximum=MAX_STRUCTURED_INTEGER)
    except ProcedureContractError as exc:
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            str(exc),
        ) from exc


def _required_identifier(payload: Mapping[str, Any], *names: str) -> str:
    for name in names:
        if name in payload and payload[name] not in (None, ""):
            try:
                return _identifier(payload[name], name)
            except ProcedureContractError as exc:
                raise _admission_error(
                    TrajectoryAdmissionReason.MALFORMED_EPISODE,
                    str(exc),
                ) from exc
    raise _admission_error(
        TrajectoryAdmissionReason.MALFORMED_EPISODE,
        f"{names[0]} is required",
    )


def _optional_identifier(payload: Mapping[str, Any], name: str) -> str:
    if name not in payload or payload[name] in (None, ""):
        return ""
    try:
        return _identifier(payload[name], name, required=False)
    except ProcedureContractError as exc:
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            str(exc),
        ) from exc


def _optional_identifiers(payload: Mapping[str, Any], name: str) -> tuple[str, ...]:
    if name not in payload or payload[name] is None:
        return ()
    try:
        return _strings(payload[name], name, identifiers=True)
    except ProcedureContractError as exc:
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            str(exc),
        ) from exc


def _episode_kind(payload: Mapping[str, Any]) -> EpisodeKind:
    raw = payload.get("episode_kind", payload.get("source_episode_kind"))
    if raw in (None, ""):
        raise _admission_error(
            TrajectoryAdmissionReason.INADMISSIBLE_SOURCE_KIND,
            "source episode kind is not admissible",
        )
    try:
        kind = _enum(raw, EpisodeKind, "episode_kind")
    except ProcedureContractError as exc:
        raise _admission_error(
            TrajectoryAdmissionReason.INADMISSIBLE_SOURCE_KIND,
            "source episode kind is not admissible",
        ) from exc
    if kind not in ADMISSIBLE_SOURCE_EPISODE_KINDS:
        raise _admission_error(
            TrajectoryAdmissionReason.INADMISSIBLE_SOURCE_KIND,
            "source episode kind is not admissible",
        )
    return kind


def _episode_bindings(payload: Mapping[str, Any]) -> ArtifactBindings:
    if "bindings" not in payload:
        raise _admission_error(
            TrajectoryAdmissionReason.STALE_RECEIPT,
            "source episode is missing current bindings",
        )
    try:
        return _nested(payload["bindings"], ArtifactBindings, "bindings")
    except ProcedureContractError as exc:
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            str(exc),
        ) from exc


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    items: list[str] = []
    for value in values:
        if value and value not in items:
            items.append(value)
    return tuple(items)


@dataclass(frozen=True)
class TrajectoryAdmissionDecision:
    """Bounded result of evaluating a source episode against the admission policy."""

    admitted: bool
    reason_code: str
    source_episode_cid: str = ""
    source_episode_kind: EpisodeKind | None = None
    evidence_cids: tuple[str, ...] = ()
    bindings: ArtifactBindings | None = None
    message: str = ""

    def __post_init__(self) -> None:
        if type(self.admitted) is not bool:
            raise TrajectoryAdmissionError("admitted must be a boolean")
        object.__setattr__(
            self,
            "reason_code",
            _identifier(self.reason_code, "reason_code", required=not self.admitted),
        )
        object.__setattr__(
            self,
            "source_episode_cid",
            _identifier(self.source_episode_cid, "source_episode_cid", required=self.admitted),
        )
        if self.source_episode_kind is not None:
            object.__setattr__(
                self,
                "source_episode_kind",
                _enum(self.source_episode_kind, EpisodeKind, "source_episode_kind"),
            )
        object.__setattr__(
            self,
            "evidence_cids",
            _strings(self.evidence_cids, "evidence_cids", identifiers=True),
        )
        if self.bindings is not None:
            object.__setattr__(
                self, "bindings", _nested(self.bindings, ArtifactBindings, "bindings")
            )
        if self.admitted and self.reason_code:
            raise TrajectoryAdmissionError("an admitted decision cannot carry a rejection reason")
        if self.admitted and not self.evidence_cids:
            raise TrajectoryAdmissionError("an admitted decision requires independent evidence")


@dataclass(frozen=True)
class TrajectoryAdmissionPolicy:
    """Fail-closed current-tree admission rules for independently validated episodes."""

    current_bindings: ArtifactBindings
    now_ms: int
    max_receipt_age_ms: int = DEFAULT_MAX_RECEIPT_AGE_MS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_bindings",
            _nested(self.current_bindings, ArtifactBindings, "current_bindings"),
        )
        object.__setattr__(
            self, "now_ms", _nonnegative_int(self.now_ms, "now_ms", maximum=MAX_STRUCTURED_INTEGER)
        )
        object.__setattr__(
            self,
            "max_receipt_age_ms",
            _nonnegative_int(
                self.max_receipt_age_ms, "max_receipt_age_ms", maximum=MAX_STRUCTURED_INTEGER
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TRAJECTORY_ADMISSION_POLICY_SCHEMA,
            "current_bindings": self.current_bindings.to_dict(),
            "now_ms": self.now_ms,
            "max_receipt_age_ms": self.max_receipt_age_ms,
            "admissible_source_kinds": tuple(
                kind.value
                for kind in sorted(ADMISSIBLE_SOURCE_EPISODE_KINDS, key=lambda item: item.value)
            ),
            "success_demonstration_source_kinds": tuple(
                kind.value
                for kind in sorted(SUCCESS_DEMONSTRATION_SOURCE_KINDS, key=lambda item: item.value)
            ),
        }

    def decide(self, episode: Any) -> TrajectoryAdmissionDecision:
        try:
            payload = _decode_source_mapping(episode)
            cleaned, _removed = self._prepare(payload)
            return self._decide_prepared(cleaned)
        except TrajectoryAdmissionError as exc:
            return TrajectoryAdmissionDecision(
                admitted=False,
                reason_code=exc.reason_code,
                message=str(exc),
            )
        except TrajectoryContractError as exc:
            return TrajectoryAdmissionDecision(
                admitted=False,
                reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
                message=str(exc),
            )

    def admit(self, episode: Any) -> TrajectoryAdmissionDecision:
        payload = _decode_source_mapping(episode)
        cleaned, _removed = self._prepare(payload)
        decision = self._decide_prepared(cleaned)
        if not decision.admitted:
            raise TrajectoryAdmissionError(
                decision.message or decision.reason_code,
                reason_code=decision.reason_code,
            )
        return decision

    def _prepare(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
        _reject_forbidden_evidence(payload)
        cleaned, removed = _redact_mapping(payload, allowed=_EPISODE_FIELDS, path="source episode")
        return cleaned, tuple(sorted(set(removed)))

    def _decide_prepared(self, payload: Mapping[str, Any]) -> TrajectoryAdmissionDecision:
        try:
            kind = _episode_kind(payload)
            bindings = _episode_bindings(payload)
            episode_cid = _required_identifier(payload, "episode_cid", "source_episode_cid")
            self._check_production_mode(payload)
            self._check_signature(payload)
            self._check_current(payload, bindings)
            self._check_freshness(payload)
            steps = payload.get("steps")
            if not isinstance(steps, Sequence) or isinstance(
                steps, (str, bytes, bytearray, memoryview)
            ):
                raise _admission_error(
                    TrajectoryAdmissionReason.MISSING_STEPS,
                    "source episode is missing ordered steps",
                )
            if not steps:
                raise _admission_error(
                    TrajectoryAdmissionReason.MISSING_STEPS,
                    "source episode is missing ordered steps",
                )
            evidence = _unique(
                (episode_cid,)
                + _optional_identifiers(payload, "admitted_evidence_cids")
                + _optional_identifiers(payload, "validation_receipt_cids")
                + _optional_identifiers(payload, "proof_receipt_cids")
            )
            return TrajectoryAdmissionDecision(
                admitted=True,
                reason_code="",
                source_episode_cid=episode_cid,
                source_episode_kind=kind,
                evidence_cids=evidence,
                bindings=bindings,
            )
        except TrajectoryAdmissionError as exc:
            return TrajectoryAdmissionDecision(
                admitted=False,
                reason_code=exc.reason_code,
                message=str(exc),
            )

    def _check_production_mode(self, payload: Mapping[str, Any]) -> None:
        if _flag(payload, "simulated"):
            raise _admission_error(
                TrajectoryAdmissionReason.SIMULATED_PRODUCTION,
                "simulated production cannot be a positive demonstration",
            )
        if _flag(payload, "pre_merge_only"):
            raise _admission_error(
                TrajectoryAdmissionReason.PRE_MERGE_ONLY_VALIDATION,
                "pre-merge-only validation cannot be a positive demonstration",
            )
        if _flag(payload, "stale"):
            raise _admission_error(
                TrajectoryAdmissionReason.STALE_RECEIPT,
                "stale receipts cannot be admitted",
            )
        source_class = _optional_identifier(payload, "source_class") or _optional_identifier(
            payload, "evidence_class"
        )
        if source_class:
            reason = _FORBIDDEN_SOURCE_CLASSES.get(_normalize_key(source_class))
            if reason is not None:
                raise _admission_error(
                    reason,
                    f"{reason.value.replace('_', ' ')} cannot be a positive demonstration",
                )
        production_mode = _optional_identifier(payload, "production_mode")
        if production_mode and production_mode not in LIVE_PRODUCTION_MODES:
            raise _admission_error(
                TrajectoryAdmissionReason.SIMULATED_PRODUCTION,
                "simulated production cannot be a positive demonstration",
            )

    def _check_signature(self, payload: Mapping[str, Any]) -> None:
        if _flag(payload, "unsigned") or (
            "signed" in payload and _flag(payload, "signed") is False
        ):
            raise _admission_error(
                TrajectoryAdmissionReason.UNSIGNED_RECEIPT,
                "unsigned receipts cannot be admitted",
            )
        signature = payload.get("signature")
        if signature in (None, ""):
            raise _admission_error(
                TrajectoryAdmissionReason.UNSIGNED_RECEIPT,
                "unsigned receipts cannot be admitted",
            )
        try:
            _identifier(signature, "signature")
        except ProcedureContractError as exc:
            raise _admission_error(
                TrajectoryAdmissionReason.UNSIGNED_RECEIPT,
                "unsigned receipts cannot be admitted",
            ) from exc

    def _check_current(self, payload: Mapping[str, Any], bindings: ArtifactBindings) -> None:
        if _flag(payload, "current", default=True) is False:
            raise _admission_error(
                TrajectoryAdmissionReason.STALE_RECEIPT,
                "stale receipts cannot be admitted",
            )
        if bindings != self.current_bindings:
            raise _admission_error(
                TrajectoryAdmissionReason.STALE_RECEIPT,
                "source episode bindings are stale for the current tree",
            )

    def _check_freshness(self, payload: Mapping[str, Any]) -> None:
        issued_at_ms = _optional_int(payload, "issued_at_ms")
        if issued_at_ms is None:
            raise _admission_error(
                TrajectoryAdmissionReason.STALE_RECEIPT,
                "source episode is missing issued_at_ms",
            )
        if issued_at_ms > self.now_ms:
            raise _admission_error(
                TrajectoryAdmissionReason.STALE_RECEIPT,
                "source episode is not yet current",
            )
        if self.now_ms - issued_at_ms > self.max_receipt_age_ms:
            raise _admission_error(
                TrajectoryAdmissionReason.STALE_RECEIPT,
                "source episode exceeds its current-tree age bound",
            )
        expires_at_ms = _optional_int(payload, "expires_at_ms")
        if expires_at_ms and self.now_ms > expires_at_ms:
            raise _admission_error(
                TrajectoryAdmissionReason.STALE_RECEIPT,
                "source episode has expired",
            )


def _reject_forbidden_evidence(payload: Mapping[str, Any]) -> None:
    for key in payload:
        reason = _FORBIDDEN_EVIDENCE_FIELDS.get(_normalize_key(key))
        if reason is not None:
            raise _admission_error(
                reason,
                f"source episode offers {reason.value.replace('_', ' ')}",
            )


def _step_status(value: Any) -> TraceEventStatus:
    try:
        if value in (None, ""):
            return TraceEventStatus.SUCCEEDED
        return _enum(value, TraceEventStatus, "status")
    except ProcedureContractError as exc:
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            str(exc),
        ) from exc


def _step_operation(value: Any) -> StepOperation:
    if isinstance(value, str) and value in FORBIDDEN_STEP_OPERATIONS:
        raise _admission_error(
            TrajectoryAdmissionReason.FORBIDDEN_OPERATION,
            "source episode names a forbidden step operation",
        )
    try:
        return _enum(value, StepOperation, "operation")
    except ProcedureContractError as exc:
        if isinstance(value, str) and value in FORBIDDEN_STEP_OPERATIONS:
            raise _admission_error(
                TrajectoryAdmissionReason.FORBIDDEN_OPERATION,
                "source episode names a forbidden step operation",
            ) from exc
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            str(exc),
        ) from exc


def _step_hole_type(payload: Mapping[str, Any], model_calls: int) -> str:
    hole_type = _optional_identifier(payload, "hole_type")
    if hole_type in FORBIDDEN_HOLE_TYPES:
        raise _admission_error(
            TrajectoryAdmissionReason.FORBIDDEN_HOLE,
            "source episode names a forbidden hole type",
        )
    if model_calls and not hole_type:
        raise _admission_error(
            TrajectoryAdmissionReason.FORBIDDEN_HOLE,
            "model calls must be attributed to a typed hole",
        )
    if hole_type:
        try:
            HoleType(hole_type)
        except ValueError as exc:
            raise _admission_error(
                TrajectoryAdmissionReason.FORBIDDEN_HOLE,
                "source episode names an unknown hole type",
            ) from exc
    return hole_type


def _build_steps(payload: Mapping[str, Any]) -> tuple[TrajectoryStep, ...]:
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(
        raw_steps, (str, bytes, bytearray, memoryview)
    ):
        raise _admission_error(
            TrajectoryAdmissionReason.MISSING_STEPS,
            "source episode is missing ordered steps",
        )
    if not raw_steps:
        raise _admission_error(
            TrajectoryAdmissionReason.MISSING_STEPS,
            "source episode is missing ordered steps",
        )
    if len(raw_steps) > MAX_STEPS:
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            "source episode exceeds the step bound",
        )
    prepared: list[tuple[int, int, Mapping[str, Any]]] = []
    for index, item in enumerate(raw_steps):
        if isinstance(item, TrajectoryStep):
            item = item.to_dict()
        if not isinstance(item, Mapping):
            raise _admission_error(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                "trajectory steps must be mappings",
            )
        sequence = item.get("sequence", index)
        try:
            sequence_value = _nonnegative_int(sequence, "sequence")
        except ProcedureContractError as exc:
            raise _admission_error(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                str(exc),
            ) from exc
        prepared.append((sequence_value, index, item))
    prepared.sort(key=lambda entry: (entry[0], entry[1]))
    sequences = [entry[0] for entry in prepared]
    if len(set(sequences)) != len(sequences):
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            "trajectory sequences must be unique",
        )
    steps: list[TrajectoryStep] = []
    for new_sequence, (_, _, item) in enumerate(prepared):
        try:
            model_calls = _nonnegative_int(item.get("model_calls", 0), "model_calls")
            input_tokens = _nonnegative_int(item.get("input_tokens", 0), "input_tokens")
            output_tokens = _nonnegative_int(item.get("output_tokens", 0), "output_tokens")
            latency_ms = _nonnegative_int(item.get("latency_ms", 0), "latency_ms")
            human_interventions = _nonnegative_int(
                item.get("human_interventions", 0), "human_interventions"
            )
            observation_cids = _strings(
                item.get("observation_cids", ()),
                "observation_cids",
                identifiers=True,
                required=True,
            )
            effect_ids = _strings(
                item.get("effect_ids", ()),
                "effect_ids",
                identifiers=True,
                required=True,
            )
            validation_receipt_cids = _strings(
                item.get("validation_receipt_cids", ()),
                "validation_receipt_cids",
                identifiers=True,
            )
        except ProcedureContractError as exc:
            message = str(exc)
            if "must not be empty" in message and "observation" in message:
                raise _admission_error(
                    TrajectoryAdmissionReason.MISSING_OBSERVATION,
                    "every normalized step requires observations",
                ) from exc
            if "must not be empty" in message and "effect" in message:
                raise _admission_error(
                    TrajectoryAdmissionReason.MISSING_EFFECT,
                    "every normalized step requires effects",
                ) from exc
            raise _admission_error(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                message,
            ) from exc
        steps.append(
            TrajectoryStep(
                sequence=new_sequence,
                operation=_step_operation(item.get("operation")),
                operation_contract=_required_identifier(item, "operation_contract"),
                initial_state_cid=_required_identifier(item, "initial_state_cid"),
                terminal_state_cid=_required_identifier(item, "terminal_state_cid"),
                observation_cids=observation_cids,
                effect_ids=effect_ids,
                validation_receipt_cids=validation_receipt_cids,
                hole_type=_step_hole_type(item, model_calls),
                model_calls=model_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                human_interventions=human_interventions,
                status=_step_status(item.get("status")),
            )
        )
    return tuple(steps)


def _outcome_status(
    payload: Mapping[str, Any],
    kind: EpisodeKind,
) -> TrajectoryTerminalStatus:
    raw = payload.get("outcome_status")
    if raw in (None, ""):
        return _DEFAULT_OUTCOME_STATUS.get(kind, TrajectoryTerminalStatus.ACCEPTED)
    try:
        return _enum(raw, TrajectoryTerminalStatus, "outcome_status")
    except ProcedureContractError as exc:
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            str(exc),
        ) from exc


def _check_kind_shape(
    kind: EpisodeKind,
    steps: tuple[TrajectoryStep, ...],
    outcome: TrajectoryOutcome,
    human_interventions: int,
) -> None:
    operations = {step.operation for step in steps}
    required = _KIND_REQUIRED_ANY_OPERATION.get(kind)
    if required and operations.isdisjoint(required):
        raise _admission_error(
            TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH,
            "source episode operations do not match the declared kind",
        )
    if kind is EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT and human_interventions < 1:
        raise _admission_error(
            TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH,
            "authorized human decisions require a human intervention",
        )
    if kind is EpisodeKind.VERIFIED_PROOF_RECEIPT and not outcome.proof_receipt_cids:
        raise _admission_error(
            TrajectoryAdmissionReason.MISSING_VALIDATION,
            "verified proof receipts require proof evidence",
        )
    if kind is EpisodeKind.FAILED_RECOVERED_EXECUTION:
        statuses = [step.status for step in steps]
        if TraceEventStatus.FAILED not in statuses:
            raise _admission_error(
                TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH,
                "failed-then-recovered episodes require a failed step",
            )
        recovered = any(
            status in {TraceEventStatus.SUCCEEDED, TraceEventStatus.ROLLED_BACK}
            for status in statuses[1:]
        )
        if not recovered:
            raise _admission_error(
                TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH,
                "failed-then-recovered episodes require a later recovered step",
            )
        if outcome.status is not TrajectoryTerminalStatus.FAILED_RECOVERED:
            raise _admission_error(
                TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH,
                "failed-then-recovered episodes require a failed_recovered terminal state",
            )
    if kind is EpisodeKind.REJECTED_TASK_RECORD:
        if outcome.status is not TrajectoryTerminalStatus.REJECTED:
            raise _admission_error(
                TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH,
                "rejected task records cannot demonstrate accepted success",
            )
        if not outcome.rejection_reason_code:
            raise _admission_error(
                TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH,
                "rejected trajectories require a typed reason",
            )
    if (
        outcome.status is TrajectoryTerminalStatus.ACCEPTED
        and kind not in SUCCESS_DEMONSTRATION_SOURCE_KINDS
    ):
        raise _admission_error(
            TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH,
            "source kind cannot demonstrate accepted success",
        )


def _build_outcome(
    payload: Mapping[str, Any],
    kind: EpisodeKind,
    steps: tuple[TrajectoryStep, ...],
) -> TrajectoryOutcome:
    status = _outcome_status(payload, kind)
    if (
        status is TrajectoryTerminalStatus.ACCEPTED
        and kind not in SUCCESS_DEMONSTRATION_SOURCE_KINDS
    ):
        raise _admission_error(
            TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH,
            "source kind cannot demonstrate accepted success",
        )
    if (
        kind is EpisodeKind.FAILED_RECOVERED_EXECUTION
        and status is not TrajectoryTerminalStatus.FAILED_RECOVERED
    ):
        raise _admission_error(
            TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH,
            "failed-then-recovered episodes require a failed_recovered terminal state",
        )
    if kind is EpisodeKind.REJECTED_TASK_RECORD and status is not TrajectoryTerminalStatus.REJECTED:
        raise _admission_error(
            TrajectoryAdmissionReason.KIND_SHAPE_MISMATCH,
            "rejected task records cannot demonstrate accepted success",
        )
    accepted = _optional_identifiers(payload, "accepted_criterion_ids")
    if status is TrajectoryTerminalStatus.ACCEPTED and not accepted:
        accepted = _optional_identifiers(payload, "objective_criterion_ids")
    step_validation = _unique(
        tuple(receipt for step in steps for receipt in step.validation_receipt_cids)
    )
    validation = _unique(step_validation + _optional_identifiers(payload, "validation_receipt_cids"))
    proof = _optional_identifiers(payload, "proof_receipt_cids")
    rejection = _optional_identifier(payload, "rejection_reason_code")
    if status is TrajectoryTerminalStatus.REJECTED and not rejection:
        rejection = DEFAULT_REJECTION_REASON
    if not validation:
        raise _admission_error(
            TrajectoryAdmissionReason.MISSING_VALIDATION,
            "normalized trajectories require validation evidence",
        )
    try:
        return TrajectoryOutcome(
            status=status,
            accepted_criterion_ids=accepted,
            validation_receipt_cids=validation,
            proof_receipt_cids=proof,
            rejection_reason_code=rejection,
        )
    except ProcedureContractError as exc:
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            str(exc),
        ) from exc


def _complete_costs(
    payload: Mapping[str, Any],
    steps: tuple[TrajectoryStep, ...],
) -> tuple[int, int, int, int]:
    step_tokens = sum(step.input_tokens + step.output_tokens for step in steps)
    step_latency = sum(step.latency_ms for step in steps)
    step_humans = sum(step.human_interventions for step in steps)
    step_cost = sum(step.model_calls + step.human_interventions for step in steps)
    raw_steps = payload.get("steps")
    if isinstance(raw_steps, Sequence) and not isinstance(
        raw_steps, (str, bytes, bytearray, memoryview)
    ):
        declared_step_cost = 0
        saw_cost = False
        for item in raw_steps:
            if isinstance(item, Mapping) and "cost_units" in item:
                saw_cost = True
                try:
                    declared_step_cost += _nonnegative_int(item["cost_units"], "cost_units")
                except ProcedureContractError as exc:
                    raise _admission_error(
                        TrajectoryAdmissionReason.INCONSISTENT_COST,
                        str(exc),
                    ) from exc
        if saw_cost:
            step_cost = declared_step_cost
    tokens = _optional_int(payload, "total_tokens")
    if tokens is None:
        tokens = step_tokens
    elif tokens != step_tokens:
        raise _admission_error(
            TrajectoryAdmissionReason.INCONSISTENT_COST,
            "trajectory token total is not denominator-preserving",
        )
    latency = _optional_int(payload, "total_latency_ms")
    if latency is None:
        latency = step_latency
    elif latency < step_latency:
        raise _admission_error(
            TrajectoryAdmissionReason.INCONSISTENT_COST,
            "trajectory latency omits step latency",
        )
    humans = _optional_int(payload, "human_interventions")
    if humans is None:
        humans = step_humans
    elif humans != step_humans:
        raise _admission_error(
            TrajectoryAdmissionReason.INCONSISTENT_COST,
            "trajectory human-intervention total is inconsistent",
        )
    cost = _optional_int(payload, "total_cost_units")
    if cost is None:
        cost = step_cost
    return cost, tokens, latency, humans


def _build_trajectory(
    payload: Mapping[str, Any],
    *,
    kind: EpisodeKind,
    bindings: ArtifactBindings,
    episode_cid: str,
) -> ExecutionTrajectory:
    steps = _build_steps(payload)
    outcome = _build_outcome(payload, kind, steps)
    cost, tokens, latency, humans = _complete_costs(payload, steps)
    _check_kind_shape(kind, steps, outcome, humans)
    try:
        trajectory = ExecutionTrajectory(
            bindings=bindings,
            source_episode_cid=episode_cid,
            source_episode_kind=kind,
            initial_abstract_state_cid=_required_identifier(payload, "initial_abstract_state_cid"),
            terminal_abstract_state_cid=_required_identifier(payload, "terminal_abstract_state_cid"),
            objective_criterion_ids=_strings(
                payload.get("objective_criterion_ids"),
                "objective_criterion_ids",
                identifiers=True,
                required=True,
            ),
            task_family_hint=_optional_identifier(payload, "task_family_hint"),
            steps=steps,
            outcome=outcome,
            total_cost_units=cost,
            total_tokens=tokens,
            total_latency_ms=latency,
            human_interventions=humans,
        )
    except ProcedureContractError as exc:
        raise _admission_error(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            str(exc),
        ) from exc
    return validate_execution_trajectory_contract(trajectory)


@dataclass(frozen=True)
class NormalizedTrajectory:
    """Candidate normalized trajectory plus its independent admission receipt."""

    trajectory: ExecutionTrajectory
    receipt: TrajectoryNormalizationReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, ExecutionTrajectory):
            raise TrajectoryAdmissionError("trajectory must be ExecutionTrajectory")
        if not isinstance(self.receipt, TrajectoryNormalizationReceipt):
            raise TrajectoryAdmissionError("receipt must be TrajectoryNormalizationReceipt")
        if self.receipt.trajectory_cid != self.trajectory.content_id:
            raise TrajectoryAdmissionError("normalization receipt does not bind the trajectory")
        if self.receipt.source_episode_cid != self.trajectory.source_episode_cid:
            raise TrajectoryAdmissionError("normalization receipt does not bind the source episode")
        validate_execution_trajectory_contract(self.trajectory)


@dataclass(frozen=True)
class TrajectoryNormalizer:
    """Admit current validated episodes and emit bounded public trajectories."""

    policy: TrajectoryAdmissionPolicy
    revision: str = TRAJECTORY_NORMALIZER_REVISION

    def __post_init__(self) -> None:
        if not isinstance(self.policy, TrajectoryAdmissionPolicy):
            raise TrajectoryAdmissionError("policy must be TrajectoryAdmissionPolicy")
        object.__setattr__(self, "revision", _identifier(self.revision, "normalizer_revision"))

    def normalize(
        self,
        episode: Any,
        *,
        emitted_at_ms: int | None = None,
    ) -> NormalizedTrajectory:
        payload = _decode_source_mapping(episode)
        cleaned, removed = self.policy._prepare(payload)
        decision = self.policy._decide_prepared(cleaned)
        if (
            not decision.admitted
            or decision.source_episode_kind is None
            or decision.bindings is None
        ):
            raise TrajectoryAdmissionError(
                decision.message or decision.reason_code,
                reason_code=decision.reason_code,
            )
        trajectory = _build_trajectory(
            cleaned,
            kind=decision.source_episode_kind,
            bindings=decision.bindings,
            episode_cid=decision.source_episode_cid,
        )
        evidence = _unique(
            decision.evidence_cids
            + tuple(receipt for step in trajectory.steps for receipt in step.validation_receipt_cids)
            + trajectory.outcome.validation_receipt_cids
            + trajectory.outcome.proof_receipt_cids
        )
        timestamp = self.policy.now_ms if emitted_at_ms is None else emitted_at_ms
        try:
            receipt = TrajectoryNormalizationReceipt(
                bindings=trajectory.bindings,
                source_episode_cid=trajectory.source_episode_cid,
                trajectory_cid=trajectory.content_id,
                admitted_evidence_cids=evidence,
                removed_field_classes=tuple(sorted(set(removed))),
                normalizer_revision=self.revision,
                emitted_at_ms=_nonnegative_int(
                    timestamp, "emitted_at_ms", maximum=MAX_STRUCTURED_INTEGER
                ),
            )
        except ProcedureContractError as exc:
            raise _admission_error(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                str(exc),
            ) from exc
        return NormalizedTrajectory(trajectory=trajectory, receipt=receipt)


def normalize_source_episode(
    episode: Any,
    *,
    policy: TrajectoryAdmissionPolicy,
    emitted_at_ms: int | None = None,
    revision: str = TRAJECTORY_NORMALIZER_REVISION,
) -> NormalizedTrajectory:
    """Admit one source episode and emit a bounded public trajectory."""

    return TrajectoryNormalizer(policy, revision=revision).normalize(
        episode, emitted_at_ms=emitted_at_ms
    )


__all__ = [
    "ADMISSIBLE_SOURCE_EPISODE_KINDS",
    "DEFAULT_MAX_RECEIPT_AGE_MS",
    "SOURCE_EPISODE_SCHEMA",
    "SUCCESS_DEMONSTRATION_SOURCE_KINDS",
    "TRAJECTORY_ADMISSION_POLICY_SCHEMA",
    "TRAJECTORY_NORMALIZER_REVISION",
    "EpisodeKind",
    "ExecutionTrajectory",
    "NormalizedTrajectory",
    "RemovedFieldClass",
    "TrajectoryAdmissionDecision",
    "TrajectoryAdmissionError",
    "TrajectoryAdmissionPolicy",
    "TrajectoryAdmissionReason",
    "TrajectoryContractError",
    "TrajectoryNormalizationReceipt",
    "TrajectoryNormalizer",
    "TrajectoryOutcome",
    "TrajectoryStep",
    "TrajectoryTerminalStatus",
    "normalize_source_episode",
    "parse_execution_trajectory",
    "validate_execution_trajectory_contract",
]
