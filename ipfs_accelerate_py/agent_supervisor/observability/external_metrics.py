"""Resource, latency, reuse and cost accounting (EAAEF-131).

Estimates are labeled as estimates.  They are never observations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final


METRICS_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/external-metrics@1"


class MetricsError(ValueError):
    """Malformed metrics record."""


@dataclass(frozen=True)
class ExternalMetrics:
    run_id: str
    task_id: str
    observed: Mapping[str, int]
    estimated: Mapping[str, int]

    def __post_init__(self) -> None:
        if not str(self.run_id).strip() or not str(self.task_id).strip():
            raise MetricsError("run_id and task_id are required")
        object.__setattr__(self, "observed", MappingProxyType(dict(self.observed)))
        object.__setattr__(self, "estimated", MappingProxyType(dict(self.estimated)))
        for key, value in {**self.observed, **self.estimated}.items():
            if int(value) < 0:
                raise MetricsError(f"{key} must be nonnegative")
        overlap = set(self.observed).intersection(self.estimated)
        if overlap:
            raise MetricsError("observed and estimated keys must not overlap")

    def as_observation(self, key: str) -> int:
        if key in self.estimated:
            raise MetricsError(f"{key} is an estimate, not an observation")
        if key not in self.observed:
            raise MetricsError(f"{key} was not observed")
        return int(self.observed[key])

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": METRICS_SCHEMA,
                "run_id": self.run_id,
                "task_id": self.task_id,
                "observed": dict(self.observed),
                "estimated": dict(self.estimated),
            }
        )
