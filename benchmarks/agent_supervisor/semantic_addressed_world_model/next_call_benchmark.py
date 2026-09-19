"""SAWM-024 next-call benchmark and deterministic baselines.

Metrics reproduce from exact fixtures. Static recall is never hidden by
ranking. Missing vector backends are typed. Results grant no runtime authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Mapping, Sequence


CASE_FAMILIES: Final[tuple[str, ...]] = (
    "resolved",
    "dynamic",
    "callback",
    "dispatch",
    "plugin",
    "reflection",
    "exception",
    "recursion",
    "async",
    "ood",
)
SPLITS: Final[tuple[str, ...]] = ("train", "dev", "held_out", "ood")


class NextCallBenchmarkError(ValueError):
    """Closed next-call benchmark contract violation."""


@dataclass(frozen=True, slots=True)
class StaticCandidateBaseline:
    name: str = "static"

    def candidates(self, current: str, catalog: Sequence[str]) -> tuple[str, ...]:
        prefix = current.rsplit(".", 1)[0] if "." in current else current
        return tuple(item for item in catalog if item.startswith(prefix) or prefix in item)


@dataclass(frozen=True, slots=True)
class CallFrequencyBaseline:
    counts: Mapping[str, int]

    def rank(self, candidates: Sequence[str]) -> tuple[str, ...]:
        return tuple(sorted(candidates, key=lambda name: (-int(self.counts.get(name, 0)), name)))


@dataclass(frozen=True, slots=True)
class LexicalRetrievalBaseline:
    def rank(self, current: str, candidates: Sequence[str]) -> tuple[str, ...]:
        needle = current.lower()
        return tuple(sorted(candidates, key=lambda name: (needle not in name.lower(), name)))


@dataclass(frozen=True, slots=True)
class VectorRetrievalBaseline:
    available: bool = False

    def rank(self, candidates: Sequence[str]) -> tuple[str, ...]:
        if not self.available:
            raise NextCallBenchmarkError("vector_backend_unavailable")
        return tuple(candidates)


@dataclass(frozen=True, slots=True)
class LinearRankerBaseline:
    def rank(self, static_hits: Sequence[str], lexical: Sequence[str]) -> tuple[str, ...]:
        seen: list[str] = []
        for name in (*static_hits, *lexical):
            if name not in seen:
                seen.append(name)
        return tuple(seen)


@dataclass(frozen=True, slots=True)
class NextCallBenchmark:
    cases: tuple[Mapping[str, Any], ...]
    catalog: tuple[str, ...]
    frequencies: Mapping[str, int]


def load_next_call_cases(path: str | Path) -> NextCallBenchmark:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = tuple(payload["cases"])
    families = {str(item["family"]) for item in cases}
    missing = [name for name in CASE_FAMILIES if name not in families]
    if missing:
        raise NextCallBenchmarkError(f"missing case families {missing}")
    splits = {str(item["split"]) for item in cases}
    if "ood" not in splits:
        raise NextCallBenchmarkError("OOD split is required")
    catalog = tuple(payload["catalog"])
    frequencies = dict(payload.get("frequencies") or {})
    return NextCallBenchmark(cases=cases, catalog=catalog, frequencies=frequencies)


def _metrics(ranks: Sequence[int], abstentions: int, n: int) -> dict[str, float | int]:
    if n <= 0:
        raise NextCallBenchmarkError("empty denominator")
    hits = {k: sum(1 for rank in ranks if 0 < rank <= k) for k in (1, 3, 5)}
    mrr = sum(1.0 / rank for rank in ranks if rank > 0) / n
    return {
        "n": n,
        "top1": hits[1] / n,
        "top3": hits[3] / n,
        "top5": hits[5] / n,
        "mrr": mrr,
        "coverage": (n - abstentions) / n,
        "abstention": abstentions / n,
    }


def run_next_call_benchmark(path: str | Path, *, vector_available: bool = False) -> dict[str, Any]:
    bench = load_next_call_cases(path)
    static = StaticCandidateBaseline()
    frequency = CallFrequencyBaseline(bench.frequencies)
    lexical = LexicalRetrievalBaseline()
    vector = VectorRetrievalBaseline(available=vector_available)
    linear = LinearRankerBaseline()
    ranks: list[int] = []
    static_recalls = 0
    abstentions = 0
    invalid_current = 0
    ood_rejected = 0
    for case in bench.cases:
        current = str(case["current"])
        gold = str(case["gold"])
        if current not in bench.catalog:
            invalid_current += 1
        candidates = static.candidates(current, bench.catalog)
        if gold in candidates:
            static_recalls += 1
        ranked = linear.rank(frequency.rank(candidates), lexical.rank(current, candidates))
        if case.get("abstain"):
            abstentions += 1
            ranks.append(0)
            continue
        if str(case["family"]) == "ood":
            ood_rejected += 1
            ranks.append(0)
            continue
        rank = ranked.index(gold) + 1 if gold in ranked else 0
        ranks.append(rank)
    if not vector_available:
        vector_status = "vector_backend_unavailable"
    else:
        vector.rank(bench.catalog)
        vector_status = "available"
    return {
        "static_recall": static_recalls / len(bench.cases),
        "ranking": _metrics(ranks, abstentions, len(bench.cases)),
        "invalid_current_target_rate": invalid_current / len(bench.cases),
        "ood_rejection": ood_rejected / len(bench.cases),
        "vector_backend": vector_status,
        "runtime_authority": False,
    }


def validate_next_call_benchmark_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if result.get("runtime_authority") is True:
        raise NextCallBenchmarkError("benchmark results grant no runtime authority")
    ranking = result.get("ranking")
    if not isinstance(ranking, Mapping) or int(ranking.get("n") or 0) < 1:
        raise NextCallBenchmarkError("ranking denominator missing")
    if "static_recall" not in result:
        raise NextCallBenchmarkError("static recall must be reported separately from ranking")
    return result
