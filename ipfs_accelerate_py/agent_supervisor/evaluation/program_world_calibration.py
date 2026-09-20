"""SAWM-030 calibration, drift, and OOD monitor.

A learned checkpoint cannot promote without held-out, calibration, and
abstention evidence. This module never writes DuckDB.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ProgramWorldCalibrationError(ValueError):
    """Closed calibration contract violation."""


@dataclass
class ProgramWorldCalibrationMonitor:
    drift: bool = False
    ood: bool = False
    ece: float = 0.0
    ood_rate: float = 0.0
    held_out_drop: float = 0.0
    ece_limit: float = 0.05
    ood_limit: float = 0.0
    drop_limit: float = 0.0

    def healthy(self) -> bool:
        return (
            not self.drift
            and not self.ood
            and self.ece <= self.ece_limit
            and self.ood_rate <= self.ood_limit
            and self.held_out_drop <= self.drop_limit
        )

    def evaluate(self, metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(metrics or {})
        ece = float(payload.get("ece", self.ece) or 0.0)
        ood_rate = float(payload.get("ood_rate", self.ood_rate) or 0.0)
        held_out_drop = float(payload.get("held_out_drop", self.held_out_drop) or 0.0)
        snapshot = ProgramWorldCalibrationMonitor(
            drift=self.drift or ece > self.ece_limit,
            ood=self.ood or ood_rate > self.ood_limit,
            ece=ece,
            ood_rate=ood_rate,
            held_out_drop=held_out_drop,
            ece_limit=self.ece_limit,
            ood_limit=self.ood_limit,
            drop_limit=self.drop_limit,
        )
        if snapshot.held_out_drop > snapshot.drop_limit:
            snapshot.drift = True
        return {
            "healthy": snapshot.healthy(),
            "drift": snapshot.drift,
            "ood": snapshot.ood,
            "ece": snapshot.ece,
            "ood_rate": snapshot.ood_rate,
            "held_out_drop": snapshot.held_out_drop,
            "promotable": False,
            "completion_authority": False,
            "reason_code": "healthy" if snapshot.healthy() else "drift_or_ood",
        }


def evaluate_program_world_calibration(
    metrics: Mapping[str, Any] | None = None,
    *,
    monitor: ProgramWorldCalibrationMonitor | None = None,
) -> dict[str, Any]:
    result = (monitor or ProgramWorldCalibrationMonitor()).evaluate(metrics)
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        mirror_work_record(
            catalog_kind="world_model",
            record_kind="world_calibration",
            record_ref=str(result.get("reason_code") or "world-calibration"),
            subject_kind="record_cid",
            subject_ref=str(result.get("reason_code") or "world-calibration"),
        )
    except Exception:
        pass
    return result
