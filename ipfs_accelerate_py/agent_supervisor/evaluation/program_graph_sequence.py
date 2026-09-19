"""SAWM-029 fair graph-sequence experiment adapters.

Missing GNN/transformer/TAGSeq backends are typed unavailable. Results do
not grant runtime or completion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class GraphSequenceError(ValueError):
    """Closed graph-sequence experiment failure."""


@dataclass(frozen=True, slots=True)
class TypedGraphNeighborhoodPacker:
    def pack(self, nodes: Sequence[str], types: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple((node, types.get(node, "unknown")) for node in nodes)


@dataclass(frozen=True, slots=True)
class SparseTypedAttentionMask:
    def mask(self, packed: Sequence[tuple[str, str]]) -> tuple[int, ...]:
        return tuple(1 if item[1] != "unknown" else 0 for item in packed)


@dataclass(frozen=True, slots=True)
class ConstrainedGraphDeltaDecoder:
    allowed: Sequence[str]

    def decode(self, ranked: Sequence[str]) -> tuple[str, ...]:
        allowed = set(self.allowed)
        return tuple(item for item in ranked if item in allowed)


@dataclass
class GraphSequenceExperiment:
    backends: Mapping[str, bool]

    def run(self, *, nodes: Sequence[str], types: Mapping[str, str], gold: str) -> dict[str, Any]:
        packer = TypedGraphNeighborhoodPacker()
        packed = packer.pack(nodes, types)
        mask = SparseTypedAttentionMask().mask(packed)
        decoder = ConstrainedGraphDeltaDecoder(allowed=nodes)
        scores: dict[str, dict[str, Any]] = {}
        for name, available in self.backends.items():
            if not available:
                scores[name] = {"status": "backend_unavailable", "hit": False}
                continue
            ranked = decoder.decode(nodes)
            scores[name] = {"status": "available", "hit": gold in ranked, "top1": ranked[0] if ranked else None}
        baseline = decoder.decode(nodes)
        return {
            "static_baseline_hit": gold in baseline,
            "encoders": scores,
            "masked_unknown": mask.count(0),
            "runtime_authority": False,
            "completion_authority": False,
        }


def run_program_graph_sequence_ablation(
    *,
    nodes: Sequence[str],
    types: Mapping[str, str],
    gold: str,
    backends: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    selected = backends or {
        "gnn": False,
        "graph_transformer": False,
        "tagseq": False,
        "linear": True,
    }
    return GraphSequenceExperiment(selected).run(nodes=nodes, types=types, gold=gold)
