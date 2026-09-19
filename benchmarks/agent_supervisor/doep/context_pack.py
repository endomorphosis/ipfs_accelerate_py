"""DOEP-066 ContextPack reuse and omission benchmark.

Reports prefix reuse, required-span coverage, and omission rates.
Benchmark numbers grant no completion authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class ContextPackBenchmarkError(ValueError):
    """Closed ContextPack benchmark contract violation."""


@dataclass(frozen=True, slots=True)
class PackCase:
    case_id: str
    required: tuple[str, ...]
    included: tuple[str, ...]
    omitted: tuple[str, ...]
    prefix: str
    reused_prefix: bool


def run_context_pack_benchmark(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise ContextPackBenchmarkError("empty benchmark")
    parsed = [
        PackCase(
            case_id=str(item["case_id"]),
            required=tuple(item.get("required") or ()),
            included=tuple(item.get("included") or ()),
            omitted=tuple(item.get("omitted") or ()),
            prefix=str(item.get("prefix") or ""),
            reused_prefix=bool(item.get("reused_prefix")),
        )
        for item in cases
    ]
    required_dropped = 0
    reuse_hits = 0
    omission_count = 0
    for case in parsed:
        if any(span in case.omitted for span in case.required):
            required_dropped += 1
        if case.reused_prefix:
            reuse_hits += 1
        omission_count += len(case.omitted)
    n = len(parsed)
    return {
        "n": n,
        "prefix_reuse_rate": reuse_hits / n,
        "omission_rate": omission_count / n,
        "required_dropped": required_dropped,
        "completion_authority": False,
    }


def validate_context_pack_benchmark(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if result.get("completion_authority") is True:
        raise ContextPackBenchmarkError("benchmark results grant no completion authority")
    if int(result.get("required_dropped") or 0) != 0:
        raise ContextPackBenchmarkError("required spans were omitted")
    return result
