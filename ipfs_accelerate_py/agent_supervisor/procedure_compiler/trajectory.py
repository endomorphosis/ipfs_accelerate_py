"""Validated trajectory ingestion, admission, and candidate normalization.

P0 ships contract validation for already-constructed, independently admitted
trajectory contracts.  This module also owns G020 ingestion: only current,
signed, independently validated source episodes may become candidate
``ExecutionTrajectory`` artifacts, and private or unbounded fields are
redacted before persistence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from .contracts import (
    FORBIDDEN_HOLE_TYPES,
    FORBIDDEN_STEP_OPERATIONS,
    MAX_STEPS,
    ArtifactBindings,
    ArtifactState,
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

NORMALIZER_REVISION: Final[str] = "trajectory-normalizer@1"
SOURCE_EPISODE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/procedure-compiler/source-episode@1"
)
ADMITTED_EVIDENCE_CLASS: Final[str] = "independently_validated_receipt"

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

_SIMULATED_PRODUCTION_MODES: Final[frozenset[str]] = frozenset(
    {"simulated", "mock", "mocked", "fixture", "synthetic"}
)
_PROMPT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prompt",
        "prompts",
        "system_prompt",
        "user_prompt",
        "private_prompt",
        "model_prompt",
        "prompt_text",
    }
)
_CHAIN_OF_THOUGHT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "cot",
        "reasoning",
        "thinking",
        "private_reasoning",
        "model_transcript",
        "scratchpad",
    }
)
_SECRET_KEYS: Final[frozenset[str]] = frozenset(
    {
        "secret",
        "secrets",
        "password",
        "api_key",
        "private_key",
        "authorization",
        "cookie",
        "session_token",
        "refresh_token",
    }
)
_CREDENTIAL_KEYS: Final[frozenset[str]] = frozenset({"credential", "credentials"})
_BODY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "body",
        "bodies",
        "source_body",
        "request_body",
        "response_body",
        "redundant_body",
        "inline_body",
        "unbounded_body",
    }
)
_LOG_KEYS: Final[frozenset[str]] = frozenset(
    {
        "log",
        "logs",
        "log_lines",
        "unbounded_log",
        "unbounded_logs",
        "raw_log",
        "debug_log",
    }
)
_ENVELOPE_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema", "contract_version", "content_id", "cid"}
)
_ALLOWED_EPISODE_FIELDS: Final[frozenset[str]] = _ENVELOPE_FIELDS | frozenset(
    {
        "episode_cid",
        "episode_kind",
        "bindings",
        "signed",
        "unsigned",
        "signature_cid",
        "current",
        "stale",
        "simulated",
        "production",
        "production_mode",
        "pre_merge_only",
        "post_merge",
        "evidence_class",
        "initial_abstract_state_cid",
        "terminal_abstract_state_cid",
        "objective_criterion_ids",
        "task_family_hint",
        "steps",
        "outcome",
        "total_cost_units",
        "total_tokens",
        "total_latency_ms",
        "human_interventions",
        "admitted_evidence_cids",
        "emitted_at_ms",
        "current_tree_id",
        "current_repository_commit",
        "proof_receipt_cids",
    }
)
_ALLOWED_STEP_FIELDS: Final[frozenset[str]] = _ENVELOPE_FIELDS | frozenset(
    {
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
    }
)
_ALLOWED_OUTCOME_FIELDS: Final[frozenset[str]] = _ENVELOPE_FIELDS | frozenset(
    {
        "status",
        "accepted_criterion_ids",
        "validation_receipt_cids",
        "proof_receipt_cids",
        "rejection_reason_code",
    }
)
_HOLE_TYPE_VALUES: Final[frozenset[str]] = frozenset(item.value for item in HoleType)
_DEFAULT_TERMINAL_STATUS: Final[dict[EpisodeKind, TrajectoryTerminalStatus]] = {
    EpisodeKind.ACCEPTED_TASK_RECEIPT: TrajectoryTerminalStatus.ACCEPTED,
    EpisodeKind.CURRENT_TREE_POST_MERGE_RECEIPT: TrajectoryTerminalStatus.ACCEPTED,
    EpisodeKind.VERIFIED_PROOF_RECEIPT: TrajectoryTerminalStatus.ACCEPTED,
    EpisodeKind.ADMITTED_TEST_RECEIPT: TrajectoryTerminalStatus.ACCEPTED,
    EpisodeKind.SUCCESSFUL_ROLLBACK_RECEIPT: TrajectoryTerminalStatus.ROLLED_BACK,
    EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT: TrajectoryTerminalStatus.ACCEPTED,
    EpisodeKind.REJECTED_TASK_RECORD: TrajectoryTerminalStatus.REJECTED,
    EpisodeKind.FAILED_RECOVERED_EXECUTION: TrajectoryTerminalStatus.FAILED_RECOVERED,
}


class TrajectoryContractError(ProcedureContractError):
    """An already-normalized trajectory violates the P0 wire contract."""


class TrajectoryAdmissionError(TrajectoryContractError):
    """A source episode is not independently admissible for normalization."""

    def __init__(self, message: str, *, reason_code: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class TrajectoryAdmissionReason(str, Enum):
    PROSE_EPISODE = "prose_episode"
    BOARD_STATUS = "board_status"
    MODEL_CONFIDENCE = "model_confidence"
    SIMULATED_PRODUCTION = "simulated_production"
    PRE_MERGE_ONLY = "pre_merge_only"
    STALE_EPISODE = "stale_episode"
    UNSIGNED_EPISODE = "unsigned_episode"
    UNKNOWN_SOURCE_KIND = "unknown_source_kind"
    MALFORMED_EPISODE = "malformed_episode"
    INCOMPLETE_FIELDS = "incomplete_fields"
    FORBIDDEN_OPERATION = "forbidden_operation"
    FORBIDDEN_HOLE = "forbidden_hole"
    SUCCESS_KIND_MISMATCH = "success_kind_mismatch"
    FLOATING_POINT = "floating_point"
    UNSUPPORTED_FIELD = "unsupported_field"


class RedactedFieldClass(str, Enum):
    PROMPT = "prompt"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SECRET = "secret"
    CREDENTIAL = "credential"
    REDUNDANT_BODY = "redundant_body"
    UNBOUNDED_LOG = "unbounded_log"


class SourceEvidenceClass(str, Enum):
    INDEPENDENTLY_VALIDATED_RECEIPT = ADMITTED_EVIDENCE_CLASS
    PROSE = "prose"
    BOARD_STATUS = "board_status"
    MODEL_CONFIDENCE = "model_confidence"
    SIMULATED = "simulated"
    PRE_MERGE_ONLY = "pre_merge_only"
    STALE = "stale"
    UNSIGNED = "unsigned"


_REFUSAL_KEYS: Final[dict[str, TrajectoryAdmissionReason]] = {
    "prose": TrajectoryAdmissionReason.PROSE_EPISODE,
    "narrative": TrajectoryAdmissionReason.PROSE_EPISODE,
    "writeup": TrajectoryAdmissionReason.PROSE_EPISODE,
    "board_status": TrajectoryAdmissionReason.BOARD_STATUS,
    "task_board": TrajectoryAdmissionReason.BOARD_STATUS,
    "task_board_status": TrajectoryAdmissionReason.BOARD_STATUS,
    "model_confidence": TrajectoryAdmissionReason.MODEL_CONFIDENCE,
    "confidence": TrajectoryAdmissionReason.MODEL_CONFIDENCE,
    "confidence_score": TrajectoryAdmissionReason.MODEL_CONFIDENCE,
    "llm_confidence": TrajectoryAdmissionReason.MODEL_CONFIDENCE,
    "confidence_class": TrajectoryAdmissionReason.MODEL_CONFIDENCE,
}

_EVIDENCE_CLASS_REFUSALS: Final[dict[str, TrajectoryAdmissionReason]] = {
    SourceEvidenceClass.PROSE.value: TrajectoryAdmissionReason.PROSE_EPISODE,
    "status": TrajectoryAdmissionReason.BOARD_STATUS,
    SourceEvidenceClass.BOARD_STATUS.value: TrajectoryAdmissionReason.BOARD_STATUS,
    "task_board": TrajectoryAdmissionReason.BOARD_STATUS,
    SourceEvidenceClass.MODEL_CONFIDENCE.value: TrajectoryAdmissionReason.MODEL_CONFIDENCE,
    "model_score": TrajectoryAdmissionReason.MODEL_CONFIDENCE,
    SourceEvidenceClass.SIMULATED.value: TrajectoryAdmissionReason.SIMULATED_PRODUCTION,
    "simulated_production": TrajectoryAdmissionReason.SIMULATED_PRODUCTION,
    SourceEvidenceClass.PRE_MERGE_ONLY.value: TrajectoryAdmissionReason.PRE_MERGE_ONLY,
    SourceEvidenceClass.STALE.value: TrajectoryAdmissionReason.STALE_EPISODE,
    SourceEvidenceClass.UNSIGNED.value: TrajectoryAdmissionReason.UNSIGNED_EPISODE,
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
    if outcome.status == TrajectoryTerminalStatus.ACCEPTED:
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
    try:
        trajectory = ExecutionTrajectory.from_dict(value)
    except TrajectoryContractError:
        raise
    except ProcedureContractError as exc:
        raise TrajectoryContractError(str(exc)) from exc
    return validate_execution_trajectory_contract(trajectory)


def _refuse(reason: TrajectoryAdmissionReason, message: str) -> None:
    raise TrajectoryAdmissionError(message, reason_code=reason.value)


def _require_bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        _refuse(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            f"{field_name} must be a boolean",
        )
    return value


def _normalize_key(key: Any) -> str:
    if not isinstance(key, str):
        _refuse(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            "episode keys must be strings",
        )
    return key.strip().lower().replace("-", "_")


def _redaction_class(key: str, allowed: frozenset[str] | None = None) -> str | None:
    normalized = _normalize_key(key)
    if allowed is not None and (normalized in allowed or key in allowed):
        return None
    if (
        normalized in _PROMPT_KEYS
        or normalized.endswith("_prompt")
        or normalized.endswith("_prompts")
    ):
        return RedactedFieldClass.PROMPT.value
    if (
        normalized in _CHAIN_OF_THOUGHT_KEYS
        or "chain_of_thought" in normalized
        or normalized.endswith("_transcript")
    ):
        return RedactedFieldClass.CHAIN_OF_THOUGHT.value
    if normalized in _CREDENTIAL_KEYS or "credential" in normalized:
        return RedactedFieldClass.CREDENTIAL.value
    if normalized in _SECRET_KEYS or any(
        marker in normalized
        for marker in (
            "secret",
            "password",
            "api_key",
            "private_key",
            "session_token",
            "refresh_token",
        )
    ):
        return RedactedFieldClass.SECRET.value
    if (
        normalized in _BODY_KEYS
        or normalized.endswith("_body")
        or normalized.endswith("_bodies")
    ):
        return RedactedFieldClass.REDUNDANT_BODY.value
    if (
        normalized in _LOG_KEYS
        or normalized.endswith("_log")
        or normalized.endswith("_logs")
    ):
        return RedactedFieldClass.UNBOUNDED_LOG.value
    return None


def _contains_float(value: Any) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, Mapping):
        return any(_contains_float(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return any(_contains_float(item) for item in value)
    return False


def _enum_or_text(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _maybe_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return {str(key): item for key, item in converted.items()}
    return value


def _canonicalize_source_episode(payload: Mapping[str, Any]) -> dict[str, Any]:
    canonical = {str(key): item for key, item in payload.items()}
    if "bindings" in canonical:
        canonical["bindings"] = _maybe_dict(canonical["bindings"])
    if "outcome" in canonical:
        canonical["outcome"] = _maybe_dict(canonical["outcome"])
    steps = canonical.get("steps")
    if isinstance(steps, Sequence) and not isinstance(
        steps, (str, bytes, bytearray, memoryview)
    ):
        canonical["steps"] = [_maybe_dict(step) for step in steps]
    return canonical


def _flag_is_asserted(payload: Mapping[str, Any], field_name: str) -> bool:
    return field_name in payload and payload[field_name] is not False


def _as_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise TrajectoryAdmissionError(
                "episode bytes must be UTF-8",
                reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
            ) from exc
    if isinstance(value, str):
        try:
            value = json.loads(
                value,
                object_pairs_hook=_closed_object,
                parse_float=_reject_float,
                parse_constant=_reject_float,
            )
        except TrajectoryContractError as exc:
            if "floating point" in str(exc):
                _refuse(
                    TrajectoryAdmissionReason.FLOATING_POINT,
                    "source episodes cannot contain floating point values",
                )
            raise TrajectoryAdmissionError(
                str(exc),
                reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
            ) from exc
        except json.JSONDecodeError as exc:
            raise TrajectoryAdmissionError(
                "source episode JSON is malformed",
                reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
            ) from exc
    if not isinstance(value, Mapping):
        _refuse(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            "source episode must be a mapping or JSON object",
        )
    return {str(key): item for key, item in value.items()}


def _scan_refusal_keys(value: Any) -> TrajectoryAdmissionReason | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = key if isinstance(key, str) else _enum_or_text(key)
            if not isinstance(key_text, str):
                _refuse(
                    TrajectoryAdmissionReason.MALFORMED_EPISODE,
                    "episode keys must be strings",
                )
            reason = _REFUSAL_KEYS.get(_normalize_key(key_text))
            if reason is not None:
                return reason
            nested = _scan_refusal_keys(item)
            if nested is not None:
                return nested
        return None
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for item in value:
            nested = _scan_refusal_keys(item)
            if nested is not None:
                return nested
    return None


def _redact(
    value: Any,
    removed: set[str],
    allowed: frozenset[str] | None = None,
) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            field_class = _redaction_class(key, allowed)
            if field_class is not None:
                removed.add(field_class)
                continue
            nested_allowed = allowed
            normalized = _normalize_key(key)
            if normalized == "steps":
                nested_allowed = _ALLOWED_STEP_FIELDS
            elif normalized == "outcome":
                nested_allowed = _ALLOWED_OUTCOME_FIELDS
            cleaned[key] = _redact(item, removed, nested_allowed)
        return cleaned
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_redact(item, removed, allowed) for item in value]
    return value


def _reject_unknown_fields(
    payload: Mapping[str, Any],
    allowed: frozenset[str],
    field_name: str,
) -> None:
    for key in payload:
        normalized = _normalize_key(key)
        if normalized in allowed or key in allowed:
            continue
        if _redaction_class(key) is not None:
            continue
        _refuse(
            TrajectoryAdmissionReason.UNSUPPORTED_FIELD,
            f"{field_name} contains unsupported field {key}",
        )


def _optional_identifier(value: Any, field_name: str) -> str:
    if value in (None, ""):
        return ""
    try:
        return _identifier(value, field_name)
    except ProcedureContractError as exc:
        raise TrajectoryAdmissionError(
            str(exc),
            reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
        ) from exc


def _required_identifier(value: Any, field_name: str) -> str:
    try:
        return _identifier(value, field_name)
    except ProcedureContractError as exc:
        raise TrajectoryAdmissionError(
            str(exc),
            reason_code=TrajectoryAdmissionReason.INCOMPLETE_FIELDS.value,
        ) from exc


def _optional_int(payload: Mapping[str, Any], field_name: str, default: int = 0) -> int:
    if field_name not in payload:
        return default
    try:
        return _nonnegative_int(payload[field_name], field_name)
    except ProcedureContractError as exc:
        raise TrajectoryAdmissionError(
            str(exc),
            reason_code=TrajectoryAdmissionReason.INCOMPLETE_FIELDS.value,
        ) from exc


def _decode_bindings(value: Any) -> ArtifactBindings:
    try:
        return _nested(value, ArtifactBindings, "bindings")
    except (ProcedureContractError, TypeError) as exc:
        raise TrajectoryAdmissionError(
            str(exc),
            reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
        ) from exc


@dataclass(frozen=True)
class TrajectoryAdmissionPolicy:
    """Fail-closed admission for independently validated source episodes."""

    current_tree_id: str = ""
    current_repository_commit: str = ""
    current_contract_revision: str = ""
    current_policy_revision: str = ""
    current_environment_id: str = ""
    require_signature: bool = True
    require_current: bool = True
    reject_simulated: bool = True
    reject_pre_merge_only: bool = True
    reject_prose: bool = True
    reject_board_status: bool = True
    reject_model_confidence: bool = True
    admissible_source_kinds: frozenset[EpisodeKind] = ADMISSIBLE_SOURCE_EPISODE_KINDS
    success_demonstration_kinds: frozenset[EpisodeKind] = SUCCESS_DEMONSTRATION_SOURCE_KINDS

    def __post_init__(self) -> None:
        for name in (
            "current_tree_id",
            "current_repository_commit",
            "current_contract_revision",
            "current_policy_revision",
            "current_environment_id",
        ):
            object.__setattr__(self, name, _optional_identifier(getattr(self, name), name))
        for name in (
            "require_signature",
            "require_current",
            "reject_simulated",
            "reject_pre_merge_only",
            "reject_prose",
            "reject_board_status",
            "reject_model_confidence",
        ):
            if type(getattr(self, name)) is not bool:
                raise TrajectoryAdmissionError(
                    f"{name} must be a boolean",
                    reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
                )
        kinds = self.admissible_source_kinds
        if not isinstance(kinds, frozenset) or not kinds:
            raise TrajectoryAdmissionError(
                "admissible_source_kinds must be a non-empty frozenset",
                reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
            )
        object.__setattr__(
            self,
            "admissible_source_kinds",
            frozenset(EpisodeKind(kind) for kind in kinds),
        )
        success = self.success_demonstration_kinds
        if not isinstance(success, frozenset):
            raise TrajectoryAdmissionError(
                "success_demonstration_kinds must be a frozenset",
                reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
            )
        object.__setattr__(
            self,
            "success_demonstration_kinds",
            frozenset(EpisodeKind(kind) for kind in success),
        )

    def admit(self, episode: Any) -> "TrajectoryAdmission":
        payload = _canonicalize_source_episode(_as_mapping(episode))
        if _contains_float(payload):
            _refuse(
                TrajectoryAdmissionReason.FLOATING_POINT,
                "source episodes cannot contain floating point values",
            )
        schema = payload.get("schema")
        if schema not in (None, "", SOURCE_EPISODE_SCHEMA):
            _refuse(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                "source episode has an unsupported schema",
            )
        if "source_episode_kind" in payload and "episode_kind" not in payload:
            _refuse(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                "already-normalized trajectories are not admitted source episodes",
            )
        refusal = _scan_refusal_keys(payload)
        if refusal is TrajectoryAdmissionReason.PROSE_EPISODE and self.reject_prose:
            _refuse(refusal, "prose or narrative cannot be admitted as a source episode")
        if refusal is TrajectoryAdmissionReason.BOARD_STATUS and self.reject_board_status:
            _refuse(refusal, "board status cannot be admitted as a source episode")
        if (
            refusal is TrajectoryAdmissionReason.MODEL_CONFIDENCE
            and self.reject_model_confidence
        ):
            _refuse(refusal, "model confidence cannot be admitted as a source episode")

        evidence_class = payload.get("evidence_class", ADMITTED_EVIDENCE_CLASS)
        if not isinstance(evidence_class, str) or not evidence_class.strip():
            _refuse(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                "evidence_class must be a non-empty string",
            )
        evidence_class = evidence_class.strip().lower().replace("-", "_")
        evidence_refusal = _EVIDENCE_CLASS_REFUSALS.get(evidence_class)
        if evidence_refusal is not None:
            _refuse(
                evidence_refusal,
                f"{evidence_class} cannot be admitted as a source episode",
            )
        if evidence_class != ADMITTED_EVIDENCE_CLASS:
            _refuse(
                TrajectoryAdmissionReason.UNKNOWN_SOURCE_KIND,
                "evidence_class is not an independently validated receipt",
            )

        if self.reject_simulated:
            if _flag_is_asserted(payload, "simulated"):
                _refuse(
                    TrajectoryAdmissionReason.SIMULATED_PRODUCTION,
                    "simulated production cannot be admitted as a source episode",
                )
            if "production" in payload and not _require_bool(payload["production"], "production"):
                _refuse(
                    TrajectoryAdmissionReason.SIMULATED_PRODUCTION,
                    "non-production episodes cannot be admitted",
                )
            mode = _enum_or_text(payload.get("production_mode", ""))
            if isinstance(mode, str) and mode.strip().lower() in _SIMULATED_PRODUCTION_MODES:
                _refuse(
                    TrajectoryAdmissionReason.SIMULATED_PRODUCTION,
                    "simulated production cannot be admitted as a source episode",
                )

        if self.reject_pre_merge_only:
            if _flag_is_asserted(payload, "pre_merge_only"):
                _refuse(
                    TrajectoryAdmissionReason.PRE_MERGE_ONLY,
                    "pre-merge-only validation cannot be a positive demonstration",
                )
            if "post_merge" in payload and not _require_bool(payload["post_merge"], "post_merge"):
                _refuse(
                    TrajectoryAdmissionReason.PRE_MERGE_ONLY,
                    "pre-merge-only validation cannot be a positive demonstration",
                )

        if self.require_current:
            if payload.get("current") is not True:
                _refuse(
                    TrajectoryAdmissionReason.STALE_EPISODE,
                    "stale episodes cannot be admitted"
                    if "current" in payload
                    else "episodes must declare they are current",
                )
            if _flag_is_asserted(payload, "stale"):
                _refuse(
                    TrajectoryAdmissionReason.STALE_EPISODE,
                    "stale episodes cannot be admitted",
                )

        if self.require_signature:
            if _flag_is_asserted(payload, "unsigned"):
                _refuse(
                    TrajectoryAdmissionReason.UNSIGNED_EPISODE,
                    "unsigned episodes cannot be admitted",
                )
            if payload.get("signed") is not True:
                _refuse(
                    TrajectoryAdmissionReason.UNSIGNED_EPISODE,
                    "unsigned episodes cannot be admitted",
                )
            if not payload.get("signature_cid"):
                _refuse(
                    TrajectoryAdmissionReason.UNSIGNED_EPISODE,
                    "unsigned episodes cannot be admitted",
                )

        try:
            kind = _enum(payload.get("episode_kind"), EpisodeKind, "episode_kind")
        except ProcedureContractError as exc:
            raise TrajectoryAdmissionError(
                "source episode kind is not admissible",
                reason_code=TrajectoryAdmissionReason.UNKNOWN_SOURCE_KIND.value,
            ) from exc
        if kind not in self.admissible_source_kinds:
            _refuse(
                TrajectoryAdmissionReason.UNKNOWN_SOURCE_KIND,
                "source episode kind is not admissible",
            )
        if (
            kind == EpisodeKind.CURRENT_TREE_POST_MERGE_RECEIPT
            and payload.get("pre_merge_only") is True
        ):
            _refuse(
                TrajectoryAdmissionReason.PRE_MERGE_ONLY,
                "pre-merge-only validation cannot be a positive demonstration",
            )

        bindings = _decode_bindings(payload.get("bindings"))
        if self.current_tree_id and bindings.tree_id != self.current_tree_id:
            _refuse(
                TrajectoryAdmissionReason.STALE_EPISODE,
                "episode tree is not the current admitted tree",
            )
        if (
            self.current_repository_commit
            and bindings.repository_commit != self.current_repository_commit
        ):
            _refuse(
                TrajectoryAdmissionReason.STALE_EPISODE,
                "episode commit is not the current admitted commit",
            )
        if (
            self.current_contract_revision
            and bindings.contract_revision != self.current_contract_revision
        ):
            _refuse(
                TrajectoryAdmissionReason.STALE_EPISODE,
                "episode contract revision is stale",
            )
        if (
            self.current_policy_revision
            and bindings.policy_revision != self.current_policy_revision
        ):
            _refuse(
                TrajectoryAdmissionReason.STALE_EPISODE,
                "episode policy revision is stale",
            )
        if (
            self.current_environment_id
            and bindings.environment_id != self.current_environment_id
        ):
            _refuse(
                TrajectoryAdmissionReason.STALE_EPISODE,
                "episode environment is stale",
            )
        declared_tree = payload.get("current_tree_id")
        if declared_tree not in (None, "") and declared_tree != bindings.tree_id:
            _refuse(
                TrajectoryAdmissionReason.STALE_EPISODE,
                "episode tree is not the current admitted tree",
            )

        removed: set[str] = set()
        redacted = _redact(payload, removed, _ALLOWED_EPISODE_FIELDS)
        if not isinstance(redacted, dict):
            _refuse(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                "redacted episode must remain a mapping",
            )
        _reject_unknown_fields(redacted, _ALLOWED_EPISODE_FIELDS, "source episode")
        steps = redacted.get("steps")
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes, bytearray)):
            _refuse(
                TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
                "source episode must contain ordered steps",
            )
        if not steps:
            _refuse(
                TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
                "source episode must contain ordered steps",
            )
        if len(steps) > MAX_STEPS:
            _refuse(
                TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
                "source episode exceeds the step bound",
            )
        cleaned_steps: list[dict[str, Any]] = []
        for step in steps:
            if not isinstance(step, Mapping):
                _refuse(
                    TrajectoryAdmissionReason.MALFORMED_EPISODE,
                    "trajectory steps must be mappings",
                )
            step_schema = step.get("schema")
            if step_schema not in (None, "", TrajectoryStep.SCHEMA):
                _refuse(
                    TrajectoryAdmissionReason.MALFORMED_EPISODE,
                    "trajectory step has an unsupported schema",
                )
            _reject_unknown_fields(step, _ALLOWED_STEP_FIELDS, "trajectory step")
            cleaned_steps.append({str(key): value for key, value in step.items()})
        outcome = redacted.get("outcome")
        if not isinstance(outcome, Mapping):
            _refuse(
                TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
                "source episode must contain a terminal outcome",
            )
        outcome_schema = outcome.get("schema")
        if outcome_schema not in (None, "", TrajectoryOutcome.SCHEMA):
            _refuse(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                "trajectory outcome has an unsupported schema",
            )
        _reject_unknown_fields(outcome, _ALLOWED_OUTCOME_FIELDS, "trajectory outcome")
        redacted["steps"] = cleaned_steps
        redacted["outcome"] = {str(key): value for key, value in outcome.items()}

        signature_cid = _required_identifier(payload.get("signature_cid"), "signature_cid")
        episode_cid = _required_identifier(payload.get("episode_cid"), "episode_cid")
        try:
            admitted_evidence = _strings(
                redacted.get("admitted_evidence_cids", ()),
                "admitted_evidence_cids",
                identifiers=True,
            )
        except ProcedureContractError as exc:
            raise TrajectoryAdmissionError(
                str(exc),
                reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
            ) from exc
        evidence = [episode_cid, signature_cid, *admitted_evidence]
        return TrajectoryAdmission(
            episode=MappingProxyType(redacted),
            source_episode_cid=episode_cid,
            source_episode_kind=kind,
            bindings=bindings,
            signature_cid=signature_cid,
            admitted_evidence_cids=tuple(dict.fromkeys(evidence)),
            removed_field_classes=tuple(
                item.value
                for item in RedactedFieldClass
                if item.value in removed
            ),
            emitted_at_ms=_optional_int(redacted, "emitted_at_ms"),
        )


@dataclass(frozen=True)
class TrajectoryAdmission:
    """Admitted, redacted source episode ready for candidate normalization."""

    episode: Mapping[str, Any]
    source_episode_cid: str
    source_episode_kind: EpisodeKind
    bindings: ArtifactBindings
    signature_cid: str
    admitted_evidence_cids: tuple[str, ...]
    removed_field_classes: tuple[str, ...]
    emitted_at_ms: int


@dataclass(frozen=True)
class TrajectoryNormalizationResult:
    """Candidate-tier normalized trajectory plus its admission receipt."""

    trajectory: ExecutionTrajectory
    receipt: TrajectoryNormalizationReceipt
    artifact_state: ArtifactState = ArtifactState.CANDIDATE

    @property
    def candidate_artifacts(self) -> tuple[ExecutionTrajectory, TrajectoryNormalizationReceipt]:
        return (self.trajectory, self.receipt)


def _ordered_steps(raw_steps: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    decorated: list[tuple[int, int, Mapping[str, Any]]] = []
    for index, step in enumerate(raw_steps):
        try:
            sequence = _nonnegative_int(step.get("sequence", index), "sequence")
        except ProcedureContractError as exc:
            raise TrajectoryAdmissionError(
                str(exc),
                reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
            ) from exc
        decorated.append((sequence, index, step))
    decorated.sort(key=lambda item: (item[0], item[1]))
    sequences = tuple(item[0] for item in decorated)
    if sequences != tuple(range(len(sequences))):
        _refuse(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            "trajectory sequences must be contiguous from zero",
        )
    return tuple(item[2] for item in decorated)


def _complete_human_interventions(
    kind: EpisodeKind,
    steps: Sequence[TrajectoryStep],
) -> tuple[TrajectoryStep, ...]:
    total = sum(step.human_interventions for step in steps)
    if kind != EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT or total:
        return tuple(steps)
    last = steps[-1]
    completed = TrajectoryStep(
        sequence=last.sequence,
        operation=last.operation,
        operation_contract=last.operation_contract,
        initial_state_cid=last.initial_state_cid,
        terminal_state_cid=last.terminal_state_cid,
        observation_cids=last.observation_cids,
        effect_ids=last.effect_ids,
        validation_receipt_cids=last.validation_receipt_cids,
        hole_type=last.hole_type,
        model_calls=last.model_calls,
        input_tokens=last.input_tokens,
        output_tokens=last.output_tokens,
        latency_ms=last.latency_ms,
        human_interventions=1,
        status=last.status,
    )
    return tuple(steps[:-1]) + (completed,)


def _closed_name(value: Any) -> str:
    text = _enum_or_text(value)
    if text in (None, ""):
        return ""
    if not isinstance(text, str):
        return ""
    return text.strip()


def _build_step(raw: Mapping[str, Any], sequence: int) -> TrajectoryStep:
    operation_value = raw.get("operation")
    operation_name = _closed_name(operation_value)
    if operation_name.upper() in FORBIDDEN_STEP_OPERATIONS:
        _refuse(
            TrajectoryAdmissionReason.FORBIDDEN_OPERATION,
            "source episode names a forbidden operation",
        )
    hole_type = raw.get("hole_type", "") or ""
    hole_name = _closed_name(hole_type)
    if hole_name.upper() in FORBIDDEN_HOLE_TYPES or (
        hole_name and hole_name not in _HOLE_TYPE_VALUES
    ):
        _refuse(
            TrajectoryAdmissionReason.FORBIDDEN_HOLE,
            "source episode names a forbidden hole type",
        )
    try:
        return TrajectoryStep(
            sequence=sequence,
            operation=_enum(operation_value, StepOperation, "operation"),
            operation_contract=_identifier(raw.get("operation_contract"), "operation_contract"),
            initial_state_cid=_identifier(raw.get("initial_state_cid"), "initial_state_cid"),
            terminal_state_cid=_identifier(raw.get("terminal_state_cid"), "terminal_state_cid"),
            observation_cids=_strings(
                raw.get("observation_cids", ()), "observation_cids", identifiers=True
            ),
            effect_ids=_strings(raw.get("effect_ids", ()), "effect_ids", identifiers=True),
            validation_receipt_cids=_strings(
                raw.get("validation_receipt_cids", ()),
                "validation_receipt_cids",
                identifiers=True,
            ),
            hole_type=_identifier(hole_name, "hole_type", required=False),
            model_calls=_optional_int(raw, "model_calls"),
            input_tokens=_optional_int(raw, "input_tokens"),
            output_tokens=_optional_int(raw, "output_tokens"),
            latency_ms=_optional_int(raw, "latency_ms"),
            human_interventions=_optional_int(raw, "human_interventions"),
            status=_enum(raw.get("status", TraceEventStatus.SUCCEEDED), TraceEventStatus, "status"),
        )
    except TrajectoryAdmissionError:
        raise
    except ProcedureContractError as exc:
        raise TrajectoryAdmissionError(
            str(exc),
            reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
        ) from exc


def _build_outcome(
    raw: Mapping[str, Any],
    kind: EpisodeKind,
    steps: Sequence[TrajectoryStep],
    extra_proof_cids: Sequence[str] = (),
    *,
    success_demonstration_kinds: frozenset[EpisodeKind] = SUCCESS_DEMONSTRATION_SOURCE_KINDS,
) -> TrajectoryOutcome:
    default_status = _DEFAULT_TERMINAL_STATUS.get(kind)
    if default_status is None:
        _refuse(
            TrajectoryAdmissionReason.UNKNOWN_SOURCE_KIND,
            "source episode kind is not admissible",
        )
    status_value = raw.get("status", default_status)
    try:
        status = _enum(status_value, TrajectoryTerminalStatus, "status")
    except ProcedureContractError as exc:
        raise TrajectoryAdmissionError(
            str(exc),
            reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
        ) from exc
    if (
        status == TrajectoryTerminalStatus.ACCEPTED
        and kind not in success_demonstration_kinds
    ):
        _refuse(
            TrajectoryAdmissionReason.SUCCESS_KIND_MISMATCH,
            "source kind cannot demonstrate accepted success",
        )
    if kind == EpisodeKind.REJECTED_TASK_RECORD and status != TrajectoryTerminalStatus.REJECTED:
        _refuse(
            TrajectoryAdmissionReason.SUCCESS_KIND_MISMATCH,
            "rejected records cannot demonstrate another terminal class",
        )
    if (
        kind == EpisodeKind.FAILED_RECOVERED_EXECUTION
        and status == TrajectoryTerminalStatus.ACCEPTED
    ):
        _refuse(
            TrajectoryAdmissionReason.SUCCESS_KIND_MISMATCH,
            "source kind cannot demonstrate accepted success",
        )
    step_validation = tuple(
        dict.fromkeys(receipt for step in steps for receipt in step.validation_receipt_cids)
    )
    try:
        validation = _strings(
            raw.get("validation_receipt_cids", step_validation),
            "validation_receipt_cids",
            identifiers=True,
        )
        proof = _strings(
            raw.get("proof_receipt_cids", extra_proof_cids),
            "proof_receipt_cids",
            identifiers=True,
        )
        accepted = _strings(
            raw.get("accepted_criterion_ids", ()),
            "accepted_criterion_ids",
            identifiers=True,
        )
    except ProcedureContractError as exc:
        raise TrajectoryAdmissionError(
            str(exc),
            reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
        ) from exc
    if status == TrajectoryTerminalStatus.ACCEPTED:
        validation = tuple(dict.fromkeys((*validation, *step_validation)))
    if extra_proof_cids:
        proof = tuple(dict.fromkeys((*proof, *extra_proof_cids)))
    rejection = raw.get("rejection_reason_code", "")
    if kind == EpisodeKind.REJECTED_TASK_RECORD and not rejection:
        rejection = "typed_rejection"
    try:
        return TrajectoryOutcome(
            status=status,
            accepted_criterion_ids=accepted,
            validation_receipt_cids=validation,
            proof_receipt_cids=proof,
            rejection_reason_code=rejection or "",
        )
    except ProcedureContractError as exc:
        raise TrajectoryAdmissionError(
            str(exc),
            reason_code=TrajectoryAdmissionReason.INCOMPLETE_FIELDS.value,
        ) from exc


def _require_category_shape(kind: EpisodeKind, steps: Sequence[TrajectoryStep]) -> None:
    operations = {step.operation for step in steps}
    if kind == EpisodeKind.CURRENT_TREE_POST_MERGE_RECEIPT and not operations.intersection(
        {StepOperation.MERGE_IN_ISOLATED_TRAIN, StepOperation.VERIFY_MERGED_TREE}
    ):
        _refuse(
            TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
            "post-merge receipts require a merge or merged-tree verification step",
        )
    if kind == EpisodeKind.SUCCESSFUL_ROLLBACK_RECEIPT and StepOperation.ROLLBACK not in operations:
        _refuse(
            TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
            "successful rollback receipts require a rollback step",
        )
    if kind == EpisodeKind.VERIFIED_PROOF_RECEIPT and StepOperation.RUN_PROOF not in operations:
        _refuse(
            TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
            "verified proof receipts require a proof step",
        )
    if kind == EpisodeKind.ADMITTED_TEST_RECEIPT and not operations.intersection(
        {StepOperation.RUN_SELECTED_TESTS, StepOperation.RUN_FULL_TEST_FALLBACK}
    ):
        _refuse(
            TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
            "admitted test receipts require a test step",
        )
    if kind == EpisodeKind.FAILED_RECOVERED_EXECUTION:
        statuses = tuple(step.status for step in steps)
        if TraceEventStatus.FAILED not in statuses:
            _refuse(
                TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
                "failed-then-recovered records require a failed step",
            )
        failure_index = statuses.index(TraceEventStatus.FAILED)
        recovered = statuses[failure_index + 1 :]
        if not any(
            status in {TraceEventStatus.SUCCEEDED, TraceEventStatus.ROLLED_BACK}
            for status in recovered
        ):
            _refuse(
                TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
                "failed-then-recovered records require a later recovered step",
            )


def _build_trajectory(
    admission: TrajectoryAdmission,
    policy: TrajectoryAdmissionPolicy | None = None,
) -> ExecutionTrajectory:
    payload = admission.episode
    ordered = _ordered_steps(payload["steps"])
    steps = _complete_human_interventions(
        admission.source_episode_kind,
        tuple(_build_step(step, sequence) for sequence, step in enumerate(ordered)),
    )
    if not any(step.observation_cids for step in steps):
        _refuse(
            TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
            "normalized trajectories require observations",
        )
    if not any(step.effect_ids for step in steps):
        _refuse(
            TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
            "normalized trajectories require effects",
        )
    if not any(step.validation_receipt_cids for step in steps):
        _refuse(
            TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
            "normalized trajectories require validation",
        )
    _require_category_shape(admission.source_episode_kind, steps)
    try:
        extra_proof = _strings(
            payload.get("proof_receipt_cids", ()),
            "proof_receipt_cids",
            identifiers=True,
        )
    except ProcedureContractError as exc:
        raise TrajectoryAdmissionError(
            str(exc),
            reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
        ) from exc
    success_kinds = (
        policy.success_demonstration_kinds
        if policy is not None
        else SUCCESS_DEMONSTRATION_SOURCE_KINDS
    )
    outcome = _build_outcome(
        payload["outcome"],
        admission.source_episode_kind,
        steps,
        extra_proof_cids=extra_proof,
        success_demonstration_kinds=success_kinds,
    )
    if (
        admission.source_episode_kind == EpisodeKind.VERIFIED_PROOF_RECEIPT
        and not outcome.proof_receipt_cids
    ):
        _refuse(
            TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
            "verified proof receipts require proof evidence",
        )
    initial = _required_identifier(
        payload.get("initial_abstract_state_cid", steps[0].initial_state_cid),
        "initial_abstract_state_cid",
    )
    terminal = _required_identifier(
        payload.get("terminal_abstract_state_cid", steps[-1].terminal_state_cid),
        "terminal_abstract_state_cid",
    )
    if steps[0].initial_state_cid != initial or steps[-1].terminal_state_cid != terminal:
        _refuse(
            TrajectoryAdmissionReason.MALFORMED_EPISODE,
            "abstract states must bound the ordered step chain",
        )
    for previous, current in zip(steps, steps[1:], strict=False):
        if previous.terminal_state_cid != current.initial_state_cid:
            _refuse(
                TrajectoryAdmissionReason.MALFORMED_EPISODE,
                "trajectory state chain is discontinuous",
            )
    step_tokens = sum(step.input_tokens + step.output_tokens for step in steps)
    step_latency = sum(step.latency_ms for step in steps)
    step_humans = sum(step.human_interventions for step in steps)
    if "total_tokens" in payload and _optional_int(payload, "total_tokens") != step_tokens:
        _refuse(
            TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
            "trajectory token total is not denominator-preserving",
        )
    if "human_interventions" in payload:
        declared_humans = _optional_int(payload, "human_interventions")
        completed_human_decision = (
            admission.source_episode_kind
            == EpisodeKind.AUTHORIZED_HUMAN_DECISION_RECEIPT
            and declared_humans == 0
            and step_humans > 0
        )
        if declared_humans != step_humans and not completed_human_decision:
            _refuse(
                TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
                "trajectory human-intervention total is inconsistent",
            )
    total_latency = _optional_int(payload, "total_latency_ms", step_latency)
    if total_latency < step_latency:
        _refuse(
            TrajectoryAdmissionReason.INCOMPLETE_FIELDS,
            "trajectory latency omits step latency",
        )
    try:
        criteria = _strings(
            payload.get("objective_criterion_ids"),
            "objective_criterion_ids",
            identifiers=True,
            required=True,
        )
    except ProcedureContractError as exc:
        raise TrajectoryAdmissionError(
            str(exc),
            reason_code=TrajectoryAdmissionReason.INCOMPLETE_FIELDS.value,
        ) from exc
    try:
        trajectory = ExecutionTrajectory(
            bindings=admission.bindings,
            source_episode_cid=admission.source_episode_cid,
            source_episode_kind=admission.source_episode_kind,
            initial_abstract_state_cid=initial,
            terminal_abstract_state_cid=terminal,
            objective_criterion_ids=criteria,
            task_family_hint=_identifier(
                payload.get("task_family_hint", ""), "task_family_hint", required=False
            ),
            steps=steps,
            outcome=outcome,
            total_cost_units=_optional_int(payload, "total_cost_units"),
            total_tokens=step_tokens,
            total_latency_ms=total_latency,
            human_interventions=step_humans,
        )
    except ProcedureContractError as exc:
        raise TrajectoryAdmissionError(
            str(exc),
            reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
        ) from exc
    try:
        return validate_execution_trajectory_contract(trajectory)
    except TrajectoryAdmissionError:
        raise
    except ProcedureContractError as exc:
        raise TrajectoryAdmissionError(
            str(exc),
            reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
        ) from exc


class TrajectoryNormalizer:
    """Admit source episodes and persist candidate normalized trajectories."""

    REVISION: Final[str] = NORMALIZER_REVISION
    SCHEMA: Final[str] = (
        "ipfs_accelerate_py/agent-supervisor/procedure-compiler/trajectory-normalizer@1"
    )

    def __init__(
        self,
        policy: TrajectoryAdmissionPolicy | None = None,
        *,
        emitted_at_ms: int = 0,
        store: dict[str, Any] | None = None,
    ) -> None:
        self.policy = policy or TrajectoryAdmissionPolicy()
        if not isinstance(self.policy, TrajectoryAdmissionPolicy):
            raise TrajectoryAdmissionError(
                "policy must be TrajectoryAdmissionPolicy",
                reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
            )
        try:
            self.emitted_at_ms = _nonnegative_int(emitted_at_ms, "emitted_at_ms")
        except ProcedureContractError as exc:
            raise TrajectoryAdmissionError(
                str(exc),
                reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
            ) from exc
        self._store: dict[str, Any] = {} if store is None else store

    def admit(self, episode: Any) -> TrajectoryAdmission:
        return self.policy.admit(episode)

    def get(self, content_id: str) -> Any:
        try:
            return self._store[content_id]
        except KeyError:
            pass
        for item in self._store.values():
            identity = getattr(item, "content_id", None)
            if identity == content_id:
                return item
            cid = getattr(item, "cid", None)
            if cid == content_id:
                return item
        raise TrajectoryContractError("normalized artifact is not in the candidate store")

    def persist(self, result: TrajectoryNormalizationResult) -> TrajectoryNormalizationResult:
        if result.artifact_state != ArtifactState.CANDIDATE:
            raise TrajectoryContractError("normalizer can persist only candidate artifacts")
        self._store[result.trajectory.content_id] = result.trajectory
        self._store[result.receipt.content_id] = result.receipt
        return result

    def normalize(
        self,
        episode: Any,
        *,
        emitted_at_ms: int | None = None,
        persist: bool = True,
    ) -> TrajectoryNormalizationResult:
        admission = (
            episode if isinstance(episode, TrajectoryAdmission) else self.policy.admit(episode)
        )
        trajectory = _build_trajectory(admission, self.policy)
        evidence = tuple(
            dict.fromkeys(
                (
                    *admission.admitted_evidence_cids,
                    *trajectory.outcome.validation_receipt_cids,
                    *trajectory.outcome.proof_receipt_cids,
                    *(
                        receipt
                        for step in trajectory.steps
                        for receipt in step.validation_receipt_cids
                    ),
                )
            )
        )
        timestamp = admission.emitted_at_ms
        if emitted_at_ms is not None:
            try:
                timestamp = _nonnegative_int(emitted_at_ms, "emitted_at_ms")
            except ProcedureContractError as exc:
                raise TrajectoryAdmissionError(
                    str(exc),
                    reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
                ) from exc
        elif timestamp == 0:
            timestamp = self.emitted_at_ms
        try:
            receipt = TrajectoryNormalizationReceipt(
                bindings=trajectory.bindings,
                source_episode_cid=admission.source_episode_cid,
                trajectory_cid=trajectory.content_id,
                admitted_evidence_cids=evidence,
                removed_field_classes=admission.removed_field_classes,
                normalizer_revision=self.REVISION,
                emitted_at_ms=timestamp,
            )
        except ProcedureContractError as exc:
            raise TrajectoryAdmissionError(
                str(exc),
                reason_code=TrajectoryAdmissionReason.MALFORMED_EPISODE.value,
            ) from exc
        result = TrajectoryNormalizationResult(trajectory=trajectory, receipt=receipt)
        if persist:
            self.persist(result)
        return result


def normalize_trajectory(
    episode: Any,
    *,
    policy: TrajectoryAdmissionPolicy | None = None,
    emitted_at_ms: int | None = None,
    persist: bool = True,
    normalizer: TrajectoryNormalizer | None = None,
) -> TrajectoryNormalizationResult:
    """Admit and normalize one source episode into candidate artifacts."""

    worker = normalizer or TrajectoryNormalizer(
        policy, emitted_at_ms=0 if emitted_at_ms is None else emitted_at_ms
    )
    return worker.normalize(episode, emitted_at_ms=emitted_at_ms, persist=persist)


__all__ = [
    "ADMISSIBLE_SOURCE_EPISODE_KINDS",
    "ADMITTED_EVIDENCE_CLASS",
    "NORMALIZER_REVISION",
    "SOURCE_EPISODE_SCHEMA",
    "SUCCESS_DEMONSTRATION_SOURCE_KINDS",
    "EpisodeKind",
    "ExecutionTrajectory",
    "RedactedFieldClass",
    "SourceEvidenceClass",
    "TrajectoryAdmission",
    "TrajectoryAdmissionError",
    "TrajectoryAdmissionPolicy",
    "TrajectoryAdmissionReason",
    "TrajectoryContractError",
    "TrajectoryNormalizationReceipt",
    "TrajectoryNormalizationResult",
    "TrajectoryNormalizer",
    "TrajectoryOutcome",
    "TrajectoryStep",
    "TrajectoryTerminalStatus",
    "normalize_trajectory",
    "parse_execution_trajectory",
    "validate_execution_trajectory_contract",
]
