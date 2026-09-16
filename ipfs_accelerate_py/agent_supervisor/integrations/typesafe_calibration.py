"""Compare TypeSafe advice to kernel/z3 outcomes. Never auto-applies skips.

Samples store family/predicted/actual/confidence only — no prompts, SMT, or
proof text. Calibration may *force* z3/kernel, never skip extra work.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    HIGH_CONFIDENCE,
    is_trap_family,
)

MAX_SAMPLES = 256
MISMATCH_FORCE_RATE = 0.10
_LOCK = threading.Lock()
_SAMPLES: list["CalibrationSample"] = []


@dataclass(frozen=True)
class CalibrationSample:
    family: str
    predicted: str
    actual: str
    confidence: float
    trap_family: bool = False

    @property
    def matched(self) -> bool:
        return str(self.predicted).casefold() == str(self.actual).casefold()

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "predicted": self.predicted,
            "actual": self.actual,
            "confidence": round(float(self.confidence), 4),
            "trap_family": self.trap_family,
            "matched": self.matched,
        }


def record_sample(
    *,
    family: str,
    predicted: str,
    actual: str,
    confidence: float = 0.0,
    trap_family: bool = False,
    smtlib: str = "",
    case_id: str = "",
) -> CalibrationSample:
    trap = bool(trap_family) or is_trap_family(
        smtlib=smtlib, case_id=case_id or family, complexity=""
    )
    sample = CalibrationSample(
        family=str(family or "unknown")[:64],
        predicted=str(predicted or "")[:32],
        actual=str(actual or "")[:32],
        confidence=max(0.0, min(1.0, float(confidence))),
        trap_family=trap,
    )
    with _LOCK:
        _SAMPLES.append(sample)
        del _SAMPLES[:-MAX_SAMPLES]
    return sample


def samples() -> tuple[CalibrationSample, ...]:
    with _LOCK:
        return tuple(_SAMPLES)


def clear_samples() -> None:
    with _LOCK:
        _SAMPLES.clear()


def mismatch_rate(*, family: str = "") -> float:
    rows = samples()
    if family:
        want = str(family).casefold()
        rows = tuple(item for item in rows if item.family.casefold() == want)
    if not rows:
        return 0.0
    misses = sum(1 for item in rows if not item.matched)
    return misses / float(len(rows))


def should_trust_skip(
    *,
    trap_family: bool = False,
    confidence: float = 0.0,
    family: str = "",
    smtlib: str = "",
    case_id: str = "",
) -> bool:
    """Recommendation only. False means force z3/kernel. Never invents a skip."""

    if trap_family or is_trap_family(
        smtlib=smtlib, case_id=case_id or family, complexity=""
    ):
        return False
    if float(confidence) < HIGH_CONFIDENCE:
        return False
    if mismatch_rate(family=family) >= MISMATCH_FORCE_RATE:
        return False
    return True


def recommend_skip_policy() -> dict[str, Any]:
    """Human-facing suggestion. ``auto_apply`` is always false."""

    rows = samples()
    families = sorted({item.family for item in rows})
    never_skip = [
        name
        for name in families
        if mismatch_rate(family=name) >= MISMATCH_FORCE_RATE
        or any(item.trap_family for item in rows if item.family == name)
    ]
    return {
        "accepted_as_authority": False,
        "auto_apply": False,
        "min_confidence": HIGH_CONFIDENCE,
        "mismatch_force_rate": MISMATCH_FORCE_RATE,
        "sample_count": len(rows),
        "never_skip_families": never_skip,
        "mismatch_rate": round(mismatch_rate(), 4),
    }


__all__ = [
    "CalibrationSample",
    "clear_samples",
    "mismatch_rate",
    "recommend_skip_policy",
    "record_sample",
    "samples",
    "should_trust_skip",
]
