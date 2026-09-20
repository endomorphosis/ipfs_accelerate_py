"""SAWM-031 program-world inference batching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SCOPE_KEYS: tuple[str, ...] = ("privacy", "tenant", "profile", "authority")


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
        if not requests:
            raise ProgramWorldBatchingError("empty batch")
        if len(requests) > self.max_batch:
            raise ProgramWorldBatchingError("batch exceeds max_batch")
        scopes = {tuple(item.get(key) for key in SCOPE_KEYS) for item in requests}
        if len(scopes) > 1:
            raise ProgramWorldBatchingError(
                "batch cannot mix privacy/tenant/profile/authority scopes"
            )
        scope = next(iter(scopes))
        tokens = 0
        for item in requests:
            tokens += int(item.get("tokens") or 0)
        return {
            "size": len(requests),
            "scope": {key: value for key, value in zip(SCOPE_KEYS, scope)},
            "resource_tokens": tokens,
            "proposal_only": True,
            "completion_authority": False,
            "advisory": True,
        }


def batch_program_world_predictions(
    requests: Sequence[Mapping[str, Any]], *, max_batch: int = 8
) -> dict[str, Any]:
    return ProgramWorldInferenceBatcher(max_batch=max_batch).batch_program_world_predictions(
        requests
    )
