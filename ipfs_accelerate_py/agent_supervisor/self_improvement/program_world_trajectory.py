"""Normalize accepted program-world trajectories for SAWM-020.

Only accepted, rights/policy-compatible trajectories enter identity.
Observational metadata, model nominations, and unaccepted traces are
excluded. Unique or ambiguous families stay episodic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Final, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
    HarnessError,
    _bool,
    _text,
    validate_opaque_cid,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload


TRAJECTORY_SCHEMA: Final[str] = "ipfs-accelerate.program-world-trajectory@1"
NORMALIZED_TRAJECTORY_SCHEMA: Final[str] = (
    "ipfs-accelerate.normalized-program-world-trajectory@1"
)
EXCLUDED_IDENTITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "ann_score",
        "similarity",
        "model_nomination",
        "provider_trace",
        "private_reasoning",
        "observed_at",
    }
)
_MAX_STEPS: Final[int] = 64


class ProgramWorldTrajectoryError(HarnessError):
    """Closed trajectory-normalization contract violation."""


def _cid(value: Any, name: str) -> str:
    return validate_opaque_cid(value, name)


def _step(value: Any, index: int) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ProgramWorldTrajectoryError(f"step {index} must be an object")
    forbidden = set(value) & EXCLUDED_IDENTITY_FIELDS
    if forbidden:
        raise ProgramWorldTrajectoryError(
            f"step {index} rejects observational fields {sorted(forbidden)}"
        )
    action = _cid(value.get("action_cid"), "action_cid")
    state = _cid(value.get("state_cid"), "state_cid")
    effect = _cid(value.get("effect_cid") or value.get("state_cid"), "effect_cid")
    return {"action_cid": action, "state_cid": state, "effect_cid": effect}


@dataclass(frozen=True, slots=True)
class NormalizedTrajectory:
    """Accepted-only normalized trajectory identity."""

    family_cid: str
    task_cid: str
    environment_cid: str
    policy_cid: str
    steps: tuple[dict[str, str], ...]
    accepted: bool
    SCHEMA: ClassVar[str] = NORMALIZED_TRAJECTORY_SCHEMA

    @property
    def trajectory_cid(self) -> str:
        return cid_for_payload(
            {
                "schema": self.SCHEMA,
                "family_cid": self.family_cid,
                "task_cid": self.task_cid,
                "environment_cid": self.environment_cid,
                "policy_cid": self.policy_cid,
                "steps": list(self.steps),
            }
        )


class ProgramWorldTrajectoryNormalizer:
    """Fail-closed normalizer. Unaccepted traces never enter identity."""

    def normalize_accepted_trajectory(self, record: Mapping[str, Any]) -> NormalizedTrajectory:
        if not isinstance(record, Mapping):
            raise ProgramWorldTrajectoryError("trajectory must be an object")
        if not _bool(record.get("accepted", False), "accepted"):
            raise ProgramWorldTrajectoryError("unaccepted trajectories cannot be normalized")
        steps_raw = record.get("steps")
        if not isinstance(steps_raw, Sequence) or isinstance(steps_raw, (str, bytes)):
            raise ProgramWorldTrajectoryError("steps must be a sequence")
        if not steps_raw:
            raise ProgramWorldTrajectoryError("accepted trajectory has no steps")
        if len(steps_raw) > _MAX_STEPS:
            raise ProgramWorldTrajectoryError("trajectory exceeds step bound")
        steps = tuple(_step(item, index) for index, item in enumerate(steps_raw))
        family_seed = tuple((step["action_cid"], step["effect_cid"]) for step in steps)
        family_cid = cid_for_payload({"schema": "family", "steps": list(family_seed)})
        return NormalizedTrajectory(
            family_cid=family_cid,
            task_cid=_cid(record.get("task_cid"), "task_cid"),
            environment_cid=_cid(record.get("environment_cid"), "environment_cid"),
            policy_cid=_cid(record.get("policy_cid"), "policy_cid"),
            steps=steps,
            accepted=True,
        )


def normalize_accepted_trajectory(record: Mapping[str, Any]) -> NormalizedTrajectory:
    return ProgramWorldTrajectoryNormalizer().normalize_accepted_trajectory(record)
