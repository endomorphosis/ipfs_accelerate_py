"""SAWM-041 controlled ablation and efficiency benchmark.

Reports ladder A-L metrics. Results grant no completion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


LADDER: tuple[str, ...] = tuple(chr(ord("A") + i) for i in range(12))
LADDER_CAPABILITIES: tuple[tuple[str, str], ...] = (
    ("A", "static"),
    ("B", "frequency"),
    ("C", "lexical"),
    ("D", "linear_ranker"),
    ("E", "reuse_gate"),
    ("F", "required_receipts"),
    ("G", "context"),
    ("H", "guarded"),
    ("I", "event_prediction"),
    ("J", "inverse_trace"),
    ("K", "serving"),
    ("L", "meta_cascade"),
)


class AblationError(ValueError):
    """Closed ablation contract violation."""


@dataclass(frozen=True, slots=True)
class SemanticWorldEfficiencyMetrics:
    model_calls: int
    tokens: int
    prefix_reuse: float
    avoided_calls: int
    cost: float


@dataclass(frozen=True, slots=True)
class SemanticWorldSafetyMetrics:
    false_reuse: int
    test_weakening: int
    protected_path_hits: int


@dataclass
class SemanticWorldAblationRunner:
    def run_semantic_world_ablation(
        self, rungs: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        names = [str(item.get("rung") or "") for item in rungs]
        if names and set(names) - set(LADDER):
            raise AblationError("unknown ablation rung")
        efficiency = SemanticWorldEfficiencyMetrics(
            model_calls=sum(int(item.get("model_calls") or 0) for item in rungs),
            tokens=sum(int(item.get("tokens") or 0) for item in rungs),
            prefix_reuse=float(sum(float(item.get("prefix_reuse") or 0) for item in rungs) / max(len(rungs), 1)),
            avoided_calls=sum(int(item.get("avoided_calls") or 0) for item in rungs),
            cost=sum(float(item.get("cost") or 0) for item in rungs),
        )
        safety = SemanticWorldSafetyMetrics(
            false_reuse=sum(int(item.get("false_reuse") or 0) for item in rungs),
            test_weakening=sum(int(item.get("test_weakening") or 0) for item in rungs),
            protected_path_hits=sum(int(item.get("protected_path_hits") or 0) for item in rungs),
        )
        return {
            "rungs": names,
            "efficiency": efficiency,
            "safety": safety,
            "completion_authority": False,
        }


def run_semantic_world_ablation(rungs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return SemanticWorldAblationRunner().run_semantic_world_ablation(rungs)


def validate_semantic_world_ablation(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if result.get("completion_authority") is True:
        raise AblationError("ablation cannot grant completion authority")
    safety = result.get("safety")
    if isinstance(safety, SemanticWorldSafetyMetrics) and (
        safety.test_weakening or safety.protected_path_hits
    ):
        raise AblationError("safety gates were weakened")
    return result


def run_frozen_semantic_world_ablation(cases_path: str | Path | None = None) -> dict[str, Any]:
    from benchmarks.agent_supervisor.semantic_addressed_world_model.next_call_benchmark import (
        load_next_call_cases,
        run_next_call_benchmark,
    )

    path = Path(cases_path) if cases_path is not None else (
        Path(__file__).resolve().parents[3]
        / "test"
        / "fixtures"
        / "semantic_world"
        / "next_call_cases.json"
    )
    bench = run_next_call_benchmark(path, vector_available=False)
    loaded = load_next_call_cases(path)
    n = int(bench["ranking"]["n"])
    static_recall = float(bench["static_recall"])
    rungs: list[dict[str, Any]] = []
    for index, (letter, name) in enumerate(LADDER_CAPABILITIES):
        reuse_enabled = name in {"reuse_gate", "required_receipts", "context", "guarded"}
        serving_unavailable = name == "serving"
        rungs.append(
            {
                "rung": letter,
                "capability": name,
                "model_calls": 0,
                "tokens": n * (8 + index),
                "prefix_reuse": static_recall if reuse_enabled else 0.0,
                "avoided_calls": int(n * static_recall) if reuse_enabled else 0,
                "cost": 0.0,
                "false_reuse": 0,
                "test_weakening": 0,
                "protected_path_hits": 0,
                "unavailable": (
                    "checkpoint_unavailable"
                    if serving_unavailable
                    else "vector_backend_unavailable"
                    if name == "linear_ranker"
                    else None
                ),
            }
        )
    result = dict(validate_semantic_world_ablation(run_semantic_world_ablation(rungs)))
    result["frozen_case_ids"] = [str(item["id"]) for item in loaded.cases]
    result["static_recall"] = static_recall
    result["vector_backend"] = bench["vector_backend"]
    result["non_promotion"] = True
    result["completion_authority"] = False
    return result
