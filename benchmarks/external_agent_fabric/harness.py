"""Reproducible external-agent fabric benchmark harness (EAAEF-150)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final


HARNESS_SCHEMA: Final[str] = "ipfs_accelerate_py/agent-supervisor/external-benchmark-harness@1"
CONFIGURATIONS: Final[tuple[str, ...]] = ("A", "B", "C", "D")


class HarnessError(ValueError):
    """Benchmark harness identity is incomplete."""


@dataclass(frozen=True)
class BenchmarkRun:
    configuration: str
    task_id: str
    repository_id: str
    authority_id: str
    image_digest: str
    model_id: str
    provider_id: str
    prover_id: str
    budget_id: str

    def __post_init__(self) -> None:
        if self.configuration not in CONFIGURATIONS:
            raise HarnessError("configuration must be A-D")
        for name in (
            "task_id",
            "repository_id",
            "authority_id",
            "image_digest",
            "model_id",
            "provider_id",
            "prover_id",
            "budget_id",
        ):
            if not str(getattr(self, name) or "").strip():
                raise HarnessError(f"{name} is required")
        if not str(self.image_digest).startswith("sha256:"):
            raise HarnessError("image_digest must be sha256:...")

    def to_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": HARNESS_SCHEMA,
                "configuration": self.configuration,
                "task_id": self.task_id,
                "repository_id": self.repository_id,
                "authority_id": self.authority_id,
                "image_digest": self.image_digest,
                "model_id": self.model_id,
                "provider_id": self.provider_id,
                "prover_id": self.prover_id,
                "budget_id": self.budget_id,
            }
        )


def matrix_for(
    *,
    task_id: str,
    repository_id: str,
    authority_id: str,
    image_digest: str,
    model_id: str,
    provider_id: str,
    prover_id: str,
    budget_id: str,
) -> tuple[BenchmarkRun, ...]:
    """Same identities across A-D; configuration letter is the only variance."""

    return tuple(
        BenchmarkRun(
            configuration=letter,
            task_id=task_id,
            repository_id=repository_id,
            authority_id=authority_id,
            image_digest=image_digest,
            model_id=model_id,
            provider_id=provider_id,
            prover_id=prover_id,
            budget_id=budget_id,
        )
        for letter in CONFIGURATIONS
    )
