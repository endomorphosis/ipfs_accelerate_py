"""SAWM-031 program-world specialist serving, hardware, and quantization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .program_world_batching import batch_program_world_predictions


class ProgramWorldServingError(ValueError):
    """Closed serving contract violation."""


def probe_gpu() -> bool:
    """Local device probe. Never trusts a caller-claimed gpu_available flag."""

    return Path("/dev/nvidia0").exists()


@dataclass(frozen=True, slots=True)
class ProgramWorldHardwareSelector:
    def select(self, request: Mapping[str, Any]) -> str:
        wants_gpu = bool(request.get("require_gpu") or request.get("device") == "gpu")
        gpu = probe_gpu()
        if wants_gpu and not gpu:
            raise ProgramWorldServingError("gpu_unavailable")
        if request.get("gpu_available") and not gpu:
            raise ProgramWorldServingError("gpu_unavailable")
        if gpu and wants_gpu:
            return "gpu"
        return "cpu"


@dataclass(frozen=True, slots=True)
class QuantizedSpecialistBinding:
    specialist: str
    quantized: bool
    checkpoint_admitted: bool


@dataclass
class ProgramWorldModelServerAdapter:
    hardware: ProgramWorldHardwareSelector | None = None

    def serve_program_world_specialist(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if request.get("privacy_denied"):
            raise ProgramWorldServingError("privacy binding denied")
        if request.get("checkpoint_admitted") is not True:
            return {
                "served": False,
                "reason_code": "checkpoint_unavailable",
                "completion_authority": False,
            }
        device = (self.hardware or ProgramWorldHardwareSelector()).select(request)
        binding = QuantizedSpecialistBinding(
            specialist=str(request.get("specialist") or "call_ranking"),
            quantized=bool(request.get("quantized")),
            checkpoint_admitted=True,
        )
        return {
            "served": True,
            "device": device,
            "specialist": binding.specialist,
            "quantized": binding.quantized,
            "proposal_only": True,
            "completion_authority": False,
        }


def serve_program_world_specialist(request: Mapping[str, Any]) -> dict[str, Any]:
    return ProgramWorldModelServerAdapter().serve_program_world_specialist(request)
