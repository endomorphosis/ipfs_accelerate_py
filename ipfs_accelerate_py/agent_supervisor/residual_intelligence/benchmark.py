"""Frozen paired residual benchmark contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .contracts import ResidualIntelligenceError, ResidualTaskFamily, required_text

MANIFEST_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-intelligence-benchmark-manifest@1"
)
CASE_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-frozen-benchmark-case@1"
PARTITIONS: Final[tuple[str, ...]] = (
    "training",
    "development",
    "held_out",
    "adversarial",
)
REQUIRED_KINDS: Final[tuple[str, ...]] = (
    "boundary",
    "negative",
    "cross_repository",
    "unknown_ood",
)


@dataclass(frozen=True)
class FrozenBenchmarkCase:
    family: ResidualTaskFamily
    partition: str
    kind: str
    case_id: str
    hidden_test: bool = False
    schema: str = CASE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", ResidualTaskFamily(self.family))
        partition = required_text(self.partition, "partition")
        if partition not in PARTITIONS:
            raise ResidualIntelligenceError(f"unknown partition: {partition}")
        object.__setattr__(self, "partition", partition)
        kind = required_text(self.kind, "kind")
        if kind not in REQUIRED_KINDS:
            raise ResidualIntelligenceError(f"unknown case kind: {kind}")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "case_id", required_text(self.case_id, "case_id"))
        if self.hidden_test and partition == "training":
            raise ResidualIntelligenceError("hidden tests cannot enter training")


@dataclass(frozen=True)
class ResidualBenchmarkManifest:
    families: tuple[ResidualTaskFamily, ...]
    partitions: tuple[str, ...]
    frozen_root: str
    schema: str = MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        families = tuple(ResidualTaskFamily(item) for item in self.families)
        if set(families) != set(ResidualTaskFamily):
            missing = sorted(item.value for item in ResidualTaskFamily if item not in families)
            raise ResidualIntelligenceError(f"benchmark missing families: {missing}")
        object.__setattr__(self, "families", families)
        object.__setattr__(self, "partitions", tuple(self.partitions))
        if tuple(self.partitions) != PARTITIONS:
            raise ResidualIntelligenceError("benchmark partitions must be exact")
        object.__setattr__(self, "frozen_root", required_text(self.frozen_root, "frozen_root"))


@dataclass(frozen=True)
class PairedBenchmarkRunner:
    def evaluate(
        self,
        manifest: ResidualBenchmarkManifest,
        cases: Sequence[FrozenBenchmarkCase],
        *,
        prior: Mapping[str, int],
        current: Mapping[str, int],
    ) -> dict[str, Any]:
        by_family = {family: [] for family in manifest.families}
        for case in cases:
            if case.hidden_test:
                continue
            by_family[case.family].append(case)
        missing = [
            family.value
            for family, items in by_family.items()
            if not items
        ]
        if missing:
            raise ResidualIntelligenceError(f"uncovered families: {missing}")
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/residual-paired-benchmark-result@1",
            "frozen_root": manifest.frozen_root,
            "prior": dict(prior),
            "current": dict(current),
            "denominators": {family.value: len(items) for family, items in by_family.items()},
            "candidate_only": True,
        }


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
