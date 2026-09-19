"""SAWM-031 program-world inference batching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class ProgramWorldBatchingError(ValueError):
    """Closed batching contract violation."""


@dataclass
class ProgramWorldInferenceBatcher:
    max_batch: int = 8
    cancelled: bool = False

    def batch_program_world_predictions(
        self, requests: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        if self.cancelled:
            raise ProgramWorldBatchingError("batch cancelled")
        if len(requests) > self.max_batch:
            raise ProgramWorldBatchingError("batch exceeds max_batch")
        return {
            "size": len(requests),
            "proposal_only": True,
            "completion_authority": False,
        }


def batch_program_world_predictions(
    requests: Sequence[Mapping[str, Any]], *, max_batch: int = 8
) -> dict[str, Any]:
    return ProgramWorldInferenceBatcher(max_batch=max_batch).batch_program_world_predictions(
        requests
    )
