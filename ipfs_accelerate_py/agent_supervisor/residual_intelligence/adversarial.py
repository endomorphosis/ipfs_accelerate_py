"""Residual adapter over the existing Adversarial Assurance authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .contracts import ResidualIntelligenceError, required_text

CAMPAIGN_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-mutant-campaign@1"
MUTANT_FAMILIES: Final[tuple[str, ...]] = (
    "family",
    "risk",
    "effect",
    "test",
    "proof",
    "cache",
    "procedure",
    "abstention",
    "injection",
    "confidence",
    "staleness",
    "quantization",
    "disagreement",
    "leakage",
    "privacy",
    "authority",
    "completion",
)


@dataclass(frozen=True)
class CriticalMutantResult:
    family: str
    escaped: bool
    receipt_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", required_text(self.family, "family"))
        if self.family not in MUTANT_FAMILIES:
            raise ResidualIntelligenceError(f"unknown mutant family: {self.family}")
        if type(self.escaped) is not bool:
            raise ResidualIntelligenceError("escaped flag must be boolean")
        object.__setattr__(self, "receipt_id", required_text(self.receipt_id, "receipt_id"))


@dataclass(frozen=True)
class ResidualMutantCampaign:
    tree_cid: str
    results: tuple[CriticalMutantResult, ...]
    schema: str = CAMPAIGN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_cid", required_text(self.tree_cid, "tree_cid"))
        object.__setattr__(self, "results", tuple(self.results))
        observed = {item.family for item in self.results}
        missing = [name for name in MUTANT_FAMILIES if name not in observed]
        if missing:
            raise ResidualIntelligenceError(f"campaign omitted mutant families: {missing}")
        if any(item.escaped for item in self.results):
            raise ResidualIntelligenceError("critical mutant escaped")

    @property
    def critical_zero_escape(self) -> bool:
        return not any(item.escaped for item in self.results)


@dataclass(frozen=True)
class ResidualAdversarialAdapter:
    def run(
        self,
        tree_cid: str,
        receipts: Sequence[Mapping[str, Any]],
    ) -> ResidualMutantCampaign:
        results = []
        for item in receipts:
            results.append(
                CriticalMutantResult(
                    family=str(item["family"]),
                    escaped=bool(item.get("escaped", False)),
                    receipt_id=str(item["receipt_id"]),
                )
            )
        return ResidualMutantCampaign(tree_cid=tree_cid, results=tuple(results))
