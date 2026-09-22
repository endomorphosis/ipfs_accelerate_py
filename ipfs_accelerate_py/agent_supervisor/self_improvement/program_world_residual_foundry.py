"""SAWM-030 residual-intelligence corpus, training, and calibration.

Checkpoints stay proposal-only. Missing corpus is training_unavailable.
Promotion cannot complete a board task.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence


def _load_calibration_module():
    path = (
        Path(__file__).resolve().parents[1] / "evaluation" / "program_world_calibration.py"
    )
    spec = importlib.util.spec_from_file_location(
        "program_world_calibration_foundry", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("calibration monitor unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_CALIBRATION = _load_calibration_module()
ProgramWorldCalibrationMonitor = _CALIBRATION.ProgramWorldCalibrationMonitor
evaluate_program_world_calibration = _CALIBRATION.evaluate_program_world_calibration


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
            result: Mapping[str, Any] = {
                "family": family,
                "abstained": True,
                "reason_code": "drift_or_ood",
                "completion_authority": False,
            }
        else:
            prepared = self.prepare_program_world_training(family)
            if prepared.get("training_unavailable"):
                result = prepared
            else:
                result = {
                    "family": family,
                    "evaluated": True,
                    "proposal_only": True,
                    "completion_authority": False,
                }
        try:
            from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
                mirror_work_record,
            )

            record_ref = str(
                result.get("family")
                or result.get("reason_code")
                or family
                or "program-world-checkpoint"
            )
            mirror_work_record(
                catalog_kind="world_model",
                record_kind="program_world_checkpoint_evaluation",
                record_ref=record_ref,
                subject_kind="record_cid",
                subject_ref=record_ref,
            )
        except Exception:
            pass
        return result

    def admit_program_world_checkpoint(self, family: str) -> ProgramWorldCheckpointAdmission:
        evaluation = self.evaluate_program_world_checkpoint(family)
        if evaluation.get("training_unavailable") or evaluation.get("abstained"):
            result = ProgramWorldCheckpointAdmission(
                family=family,
                admitted=False,
                reason_code=str(evaluation.get("reason_code") or "not_evaluated"),
            )
        else:
            result = ProgramWorldCheckpointAdmission(
                family=family,
                admitted=True,
                reason_code="proposal_only_checkpoint",
            )
        try:
            from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
                mirror_work_record,
            )

            record_ref = str(result.family or result.reason_code or "program-world-checkpoint-admission")
            mirror_work_record(
                catalog_kind="world_model",
                record_kind="program_world_checkpoint_admission",
                record_ref=record_ref,
                subject_kind="record_cid",
                subject_ref=record_ref,
            )
        except Exception:
            pass
        return result


def prepare_program_world_training(
    family: str, *, foundry: ProgramWorldResidualFoundry | None = None
) -> Mapping[str, Any]:
    return (foundry or ProgramWorldResidualFoundry()).prepare_program_world_training(family)


def evaluate_program_world_checkpoint(
    family: str, *, foundry: ProgramWorldResidualFoundry | None = None
) -> Mapping[str, Any]:
    evaluation = (foundry or ProgramWorldResidualFoundry()).evaluate_program_world_checkpoint(family)
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        checkpoint_id = evaluation.get("checkpoint_id") if isinstance(evaluation, Mapping) else ""
        record_ref = str(checkpoint_id or family or "program-world-checkpoint")
        mirror_work_record(
            catalog_kind="world_model",
            record_kind="program_world_checkpoint_evaluation",
            record_ref=str(record_ref),
            subject_kind="record_cid",
            subject_ref=str(record_ref),
        )
    except Exception:
        pass
    return evaluation


def admit_program_world_checkpoint(
    family: str, *, foundry: ProgramWorldResidualFoundry | None = None
) -> ProgramWorldCheckpointAdmission:
    admission = (foundry or ProgramWorldResidualFoundry()).admit_program_world_checkpoint(family)
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        record_ref = str(
            getattr(admission, "checkpoint_id", "")
            or getattr(admission, "content_id", "")
            or family
            or "program-world-checkpoint-admission"
        )
        mirror_work_record(
            catalog_kind="world_model",
            record_kind="program_world_checkpoint_admission",
            record_ref=record_ref,
            subject_kind="record_cid",
            subject_ref=record_ref,
        )
    except Exception:
        pass
    return admission
