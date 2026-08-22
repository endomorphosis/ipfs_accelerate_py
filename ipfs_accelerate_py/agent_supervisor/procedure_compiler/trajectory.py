"""P0 wire ownership and contract validation for execution trajectories.

This module intentionally does not ingest or normalize historical episodes.
PCPC-009 owns that later capability.  The helpers here only validate already
constructed, independently admitted trajectory contracts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final

from .contracts import (
    EpisodeKind,
    ExecutionTrajectory,
    HoleType,
    ProcedureContractError,
    TrajectoryNormalizationReceipt,
    TrajectoryOutcome,
    TrajectoryStep,
    TrajectoryTerminalStatus,
)


class TrajectoryContractError(ProcedureContractError):
    """An already-normalized trajectory violates the P0 wire contract."""


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


__all__ = [
    "ADMISSIBLE_SOURCE_EPISODE_KINDS",
    "SUCCESS_DEMONSTRATION_SOURCE_KINDS",
    "EpisodeKind",
    "ExecutionTrajectory",
    "TrajectoryContractError",
    "TrajectoryNormalizationReceipt",
    "TrajectoryOutcome",
    "TrajectoryStep",
    "TrajectoryTerminalStatus",
    "parse_execution_trajectory",
    "validate_execution_trajectory_contract",
]
