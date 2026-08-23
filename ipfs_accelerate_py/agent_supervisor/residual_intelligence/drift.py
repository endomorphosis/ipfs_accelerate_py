"""Expert drift, demotion, and revocation. Models cannot mutate state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .contracts import ResidualIntelligenceError, required_text

DRIFT_EVENT_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-drift-event@1"


class ExpertState(str, Enum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    PROMOTED = "promoted"
    DEGRADED = "degraded"
    STALE = "stale"
    REVOKED = "revoked"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class DriftDisposition(str, Enum):
    WIDER_ABSTENTION = "wider_abstention"
    SHADOW_ONLY = "shadow_only"
    DEMOTE = "demote"
    REVOKE = "revoke"
    REEVALUATE = "reevaluate"
    RETRAINING_PROPOSAL = "retraining_proposal"


@dataclass(frozen=True)
class DriftEvent:
    expert_id: str
    signal: str
    current: bool
    schema: str = DRIFT_EVENT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "expert_id", required_text(self.expert_id, "expert_id"))
        object.__setattr__(self, "signal", required_text(self.signal, "signal"))
        if type(self.current) is not bool:
            raise ResidualIntelligenceError("drift current flag must be boolean")
        if not self.current:
            raise ResidualIntelligenceError("stale drift evidence cannot change state")


@dataclass(frozen=True)
class ExpertDriftMonitor:
    state: ExpertState = ExpertState.CANDIDATE

    def apply(self, event: DriftEvent) -> tuple[ExpertState, DriftDisposition]:
        if self.state in {ExpertState.REVOKED, ExpertState.STALE} and event.signal != "authorized_cas_restore":
            return self.state if self.state is ExpertState.REVOKED else ExpertState.STALE, DriftDisposition.SHADOW_ONLY
        mapping = {
            "false_accept": (ExpertState.REVOKED, DriftDisposition.REVOKE),
            "calibration_group_change": (ExpertState.SHADOW, DriftDisposition.WIDER_ABSTENTION),
            "hardware_change": (ExpertState.DEGRADED, DriftDisposition.REEVALUATE),
            "quantization_drift": (ExpertState.SHADOW, DriftDisposition.REEVALUATE),
            "validation_fail": (ExpertState.DEGRADED, DriftDisposition.DEMOTE),
            "family_contract": (ExpertState.REJECTED, DriftDisposition.REVOKE),
            "authorized_cas_restore": (ExpertState.CANDIDATE, DriftDisposition.REEVALUATE),
        }
        if event.signal not in mapping:
            raise ResidualIntelligenceError(f"unknown drift signal: {event.signal}")
        return mapping[event.signal]

    def routable(self, state: ExpertState) -> bool:
        return state in {ExpertState.PROMOTED, ExpertState.CANDIDATE, ExpertState.SHADOW, ExpertState.DEGRADED} and state not in {
            ExpertState.STALE,
            ExpertState.REVOKED,
            ExpertState.REJECTED,
        }
