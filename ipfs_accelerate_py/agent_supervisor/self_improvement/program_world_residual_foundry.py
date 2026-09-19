"""SAWM-030 residual-intelligence corpus, training, and calibration.

Checkpoints stay proposal-only. Missing corpus is training_unavailable.
Promotion cannot complete a board task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Sequence


FAMILIES: Final[tuple[str, ...]] = (
    "call_ranking",
    "event_state",
    "inverse_trace",
    "repair_operator",
    "graph_delta",
)


class ResidualFoundryError(ValueError):
    """Closed residual-foundry contract violation."""


@dataclass(frozen=True, slots=True)
class ProgramWorldTaskFamily:
    name: str
    corpus_admitted: bool
    split_frozen: bool


@dataclass(frozen=True, slots=True)
class ProgramWorldCheckpointAdmission:
    family: str
    admitted: bool
    reason_code: str
    proposal_only: bool = True
    completion_authority: bool = False


@dataclass
class ProgramWorldCalibrationMonitor:
    drift: bool = False
    ood: bool = False

    def healthy(self) -> bool:
        return not self.drift and not self.ood


@dataclass
class ProgramWorldResidualFoundry:
    families: dict[str, ProgramWorldTaskFamily] = field(default_factory=dict)
    monitor: ProgramWorldCalibrationMonitor = field(
        default_factory=ProgramWorldCalibrationMonitor
    )

    def prepare_program_world_training(self, family: str) -> Mapping[str, Any]:
        spec = self.families.get(family)
        if spec is None or not spec.corpus_admitted or not spec.split_frozen:
            return {
                "family": family,
                "training_unavailable": True,
                "reason_code": "corpus_or_split_unavailable",
                "completion_authority": False,
            }
        return {
            "family": family,
            "training_unavailable": False,
            "proposal_only": True,
            "completion_authority": False,
        }

    def evaluate_program_world_checkpoint(self, family: str) -> Mapping[str, Any]:
        if not self.monitor.healthy():
            return {
                "family": family,
                "abstained": True,
                "reason_code": "drift_or_ood",
                "completion_authority": False,
            }
        prepared = self.prepare_program_world_training(family)
        if prepared.get("training_unavailable"):
            return prepared
        return {
            "family": family,
            "evaluated": True,
            "proposal_only": True,
            "completion_authority": False,
        }

    def admit_program_world_checkpoint(self, family: str) -> ProgramWorldCheckpointAdmission:
        evaluation = self.evaluate_program_world_checkpoint(family)
        if evaluation.get("training_unavailable") or evaluation.get("abstained"):
            return ProgramWorldCheckpointAdmission(
                family=family,
                admitted=False,
                reason_code=str(evaluation.get("reason_code") or "not_evaluated"),
            )
        return ProgramWorldCheckpointAdmission(
            family=family,
            admitted=True,
            reason_code="proposal_only_checkpoint",
        )


def prepare_program_world_training(
    family: str, *, foundry: ProgramWorldResidualFoundry | None = None
) -> Mapping[str, Any]:
    return (foundry or ProgramWorldResidualFoundry()).prepare_program_world_training(family)


def evaluate_program_world_checkpoint(
    family: str, *, foundry: ProgramWorldResidualFoundry | None = None
) -> Mapping[str, Any]:
    return (foundry or ProgramWorldResidualFoundry()).evaluate_program_world_checkpoint(family)


def admit_program_world_checkpoint(
    family: str, *, foundry: ProgramWorldResidualFoundry | None = None
) -> ProgramWorldCheckpointAdmission:
    return (foundry or ProgramWorldResidualFoundry()).admit_program_world_checkpoint(family)
