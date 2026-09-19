"""SAWM-030 calibration monitor. No evaluation-package side imports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProgramWorldCalibrationMonitor:
    drift: bool = False
    ood: bool = False

    def healthy(self) -> bool:
        return not self.drift and not self.ood
