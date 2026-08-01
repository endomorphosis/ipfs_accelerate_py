"""Privacy-safe session reporting for proof-backed pytest reuse.

The reporting contract intentionally contains aggregates only.  Node ids,
paths, parameter values, receipt bodies, exception text, and test output are
never accepted into a metrics snapshot, including xdist worker snapshots.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

PROOF_REUSE_METRICS_INTERFACE: Final = "ProofReuseMetrics@1"
PROOF_REUSE_METRICS_SCHEMA: Final = (
    "ipfs_accelerate_py/testing/proof-reuse-metrics@1"
)
MAX_REASON_CODES: Final = 128
MAX_REASON_LENGTH: Final = 96
MAX_COUNTER_VALUE: Final = (1 << 63) - 1


class ProofReuseOutcome(str, Enum):
    """Closed outcome vocabulary used by local and xdist reporters."""

    PREDICTED = "predicted"
    VERIFIED = "verified"
    SKIPPED = "skipped"
    EXECUTED = "executed"
    DEFERRED = "deferred"
    DEGRADED = "degraded"


_OUTCOMES: Final = tuple(outcome.value for outcome in ProofReuseOutcome)
_SAFE_REASON_CHARS: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyz0123456789_:-."
)
_SAFE_REASON_CODES: Final = frozenset(
    {
        "absence_fail_open_to_run",
        "cache_unavailable",
        "candidate_integrity_failed",
        "candidate_missing",
        "certificate_deferred",
        "certificate_non_attested",
        "certificate_provider_unavailable",
        "cid_provider_unavailable",
        "circuit_unavailable",
        "coordination_unavailable",
        "deferred_issuer_unavailable",
        "eligibility_denied",
        "exception_fail_open_to_run",
        "execution_key_mismatch",
        "expired_or_revoked",
        "illegal_authority",
        "incomplete_trace",
        "internal_error_fail_open_to_run",
        "invalidation",
        "issuer_revoked",
        "key_unavailable",
        "lookup_decision_invalid",
        "lookup_hit",
        "malformed_artifact",
        "mode_off",
        "mode_shadow",
        "mode_write_only",
        "non_reusable",
        "over_budget",
        "plugin_unavailable",
        "policy_mismatch",
        "private_material",
        "proof_cache_hit",
        "proof_verified",
        "publication_failed",
        "publication_intent_disagrees",
        "publication_intent_invalid",
        "publication_over_budget",
        "real_execution",
        "receipt_mismatch",
        "reuse_disabled",
        "runtime_hook_failed",
        "runtime_registration_failed",
        "timeout",
        "trust_policy_rejected",
        "unknown",
        "unsupported",
        "verifier_unavailable",
        "worker_crash",
        "worker_output_missing",
        "worker_output_rejected",
    }
)


def _safe_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("metric counts must be integers")
    if value < 0 or value > MAX_COUNTER_VALUE:
        raise ValueError("metric count is out of bounds")
    return value


def _safe_reason(value: Any) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        return ""
    value = value.strip().lower()
    if not value or len(value) > MAX_REASON_LENGTH:
        return ""
    if any(character not in _SAFE_REASON_CHARS for character in value):
        return ""
    if value not in _SAFE_REASON_CODES:
        return ""
    return value


def _safe_packet_id(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        return ""
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        return ""
    return normalized


def _safe_latency(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("latency must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or result > 86_400_000:
        raise ValueError("latency is out of bounds")
    return result


@dataclass(frozen=True)
class ProofReuseMetricsSnapshot:
    """Immutable, JSON-safe aggregate suitable for worker transport."""

    counts: Mapping[str, int]
    reasons: Mapping[str, int]
    verify_latency_ms: float = 0.0
    execution_latency_ms: float = 0.0
    bytes_read: int = 0
    bytes_written: int = 0

    @property
    def interface(self) -> str:
        return PROOF_REUSE_METRICS_INTERFACE

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_REUSE_METRICS_SCHEMA,
            "interface": PROOF_REUSE_METRICS_INTERFACE,
            "counts": {name: int(self.counts.get(name, 0)) for name in _OUTCOMES},
            "reasons": dict(sorted(self.reasons.items())),
            "verify_latency_ms": self.verify_latency_ms,
            "execution_latency_ms": self.execution_latency_ms,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofReuseMetricsSnapshot":
        if not isinstance(payload, Mapping):
            raise ValueError("metrics snapshot must be a mapping")
        allowed = {
            "schema",
            "interface",
            "counts",
            "reasons",
            "verify_latency_ms",
            "execution_latency_ms",
            "bytes_read",
            "bytes_written",
        }
        if set(payload) - allowed:
            raise ValueError("metrics snapshot contains private or unknown fields")
        if payload.get("schema") != PROOF_REUSE_METRICS_SCHEMA:
            raise ValueError("metrics snapshot schema mismatch")
        if payload.get("interface") != PROOF_REUSE_METRICS_INTERFACE:
            raise ValueError("metrics snapshot interface mismatch")
        raw_counts = payload.get("counts")
        raw_reasons = payload.get("reasons")
        if not isinstance(raw_counts, Mapping) or not isinstance(raw_reasons, Mapping):
            raise ValueError("metrics counters must be mappings")
        if set(raw_counts) - set(_OUTCOMES):
            raise ValueError("metrics snapshot contains an unknown outcome")
        counts = {name: _safe_count(raw_counts.get(name, 0)) for name in _OUTCOMES}
        if len(raw_reasons) > MAX_REASON_CODES:
            raise ValueError("metrics snapshot has too many reason codes")
        reasons: dict[str, int] = {}
        for raw_reason, raw_count in raw_reasons.items():
            reason = _safe_reason(raw_reason)
            if not reason or reason != raw_reason:
                raise ValueError("metrics snapshot has an unsafe reason code")
            reasons[reason] = _safe_count(raw_count)
        return cls(
            counts=counts,
            reasons=reasons,
            verify_latency_ms=_safe_latency(payload.get("verify_latency_ms", 0)),
            execution_latency_ms=_safe_latency(
                payload.get("execution_latency_ms", 0)
            ),
            bytes_read=_safe_count(payload.get("bytes_read", 0)),
            bytes_written=_safe_count(payload.get("bytes_written", 0)),
        )


class ProofReuseSessionMetrics:
    """Thread-safe aggregate with explicit proof-reuse outcome dimensions."""

    interface = PROOF_REUSE_METRICS_INTERFACE

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._reasons: Counter[str] = Counter()
        self._verify_latency_ms = 0.0
        self._execution_latency_ms = 0.0
        self._bytes_read = 0
        self._bytes_written = 0
        self._merged_packets: set[str] = set()
        self._lock = threading.RLock()

    def record(
        self,
        outcome: ProofReuseOutcome | str,
        *,
        count: int = 1,
        reason_code: Any = "",
        latency_ms: int | float = 0,
        bytes_read: int = 0,
        bytes_written: int = 0,
    ) -> None:
        name = outcome.value if isinstance(outcome, ProofReuseOutcome) else str(outcome)
        if name not in _OUTCOMES:
            raise ValueError("unknown proof reuse outcome")
        increment = _safe_count(count)
        latency = _safe_latency(latency_ms)
        read_count = _safe_count(bytes_read)
        written_count = _safe_count(bytes_written)
        reason = _safe_reason(reason_code)
        with self._lock:
            self._counts[name] = min(
                MAX_COUNTER_VALUE, self._counts[name] + increment
            )
            if reason:
                if reason in self._reasons or len(self._reasons) < MAX_REASON_CODES:
                    self._reasons[reason] = min(
                        MAX_COUNTER_VALUE, self._reasons[reason] + increment
                    )
            if name == ProofReuseOutcome.VERIFIED.value:
                self._verify_latency_ms += latency
            elif name == ProofReuseOutcome.EXECUTED.value:
                self._execution_latency_ms += latency
            self._bytes_read = min(
                MAX_COUNTER_VALUE, self._bytes_read + read_count
            )
            self._bytes_written = min(
                MAX_COUNTER_VALUE, self._bytes_written + written_count
            )

    def predicted(self, **kwargs: Any) -> None:
        self.record(ProofReuseOutcome.PREDICTED, **kwargs)

    def verified(self, **kwargs: Any) -> None:
        self.record(ProofReuseOutcome.VERIFIED, **kwargs)

    def skipped(self, **kwargs: Any) -> None:
        self.record(ProofReuseOutcome.SKIPPED, **kwargs)

    def executed(self, **kwargs: Any) -> None:
        self.record(ProofReuseOutcome.EXECUTED, **kwargs)

    def deferred(self, **kwargs: Any) -> None:
        self.record(ProofReuseOutcome.DEFERRED, **kwargs)

    def degraded(self, **kwargs: Any) -> None:
        self.record(ProofReuseOutcome.DEGRADED, **kwargs)

    def count(self, outcome: ProofReuseOutcome | str) -> int:
        name = outcome.value if isinstance(outcome, ProofReuseOutcome) else str(outcome)
        with self._lock:
            return int(self._counts.get(name, 0))

    @property
    def counts(self) -> Mapping[str, int]:
        return self.snapshot().counts

    @property
    def reasons(self) -> Mapping[str, int]:
        return self.snapshot().reasons

    def snapshot(self) -> ProofReuseMetricsSnapshot:
        with self._lock:
            return ProofReuseMetricsSnapshot(
                counts={name: int(self._counts.get(name, 0)) for name in _OUTCOMES},
                reasons=dict(sorted(self._reasons.items())),
                verify_latency_ms=round(self._verify_latency_ms, 3),
                execution_latency_ms=round(self._execution_latency_ms, 3),
                bytes_read=self._bytes_read,
                bytes_written=self._bytes_written,
            )

    def merge(
        self,
        snapshot: ProofReuseMetricsSnapshot | Mapping[str, Any],
        *,
        packet_id: str = "",
    ) -> bool:
        """Merge one worker aggregate once; malformed/private data is rejected."""

        if isinstance(snapshot, Mapping):
            snapshot = ProofReuseMetricsSnapshot.from_dict(snapshot)
        if not isinstance(snapshot, ProofReuseMetricsSnapshot):
            raise TypeError("snapshot must be ProofReuseMetricsSnapshot")
        safe_packet_id = _safe_packet_id(packet_id)
        with self._lock:
            if safe_packet_id and safe_packet_id in self._merged_packets:
                return False
            for name in _OUTCOMES:
                self._counts[name] = min(
                    MAX_COUNTER_VALUE,
                    self._counts[name] + _safe_count(snapshot.counts.get(name, 0)),
                )
            for reason, count in snapshot.reasons.items():
                safe_reason = _safe_reason(reason)
                if not safe_reason or safe_reason != reason:
                    raise ValueError("unsafe reason code")
                if (
                    safe_reason in self._reasons
                    or len(self._reasons) < MAX_REASON_CODES
                ):
                    self._reasons[safe_reason] = min(
                        MAX_COUNTER_VALUE,
                        self._reasons[safe_reason] + _safe_count(count),
                    )
            self._verify_latency_ms += _safe_latency(snapshot.verify_latency_ms)
            self._execution_latency_ms += _safe_latency(
                snapshot.execution_latency_ms
            )
            self._bytes_read = min(
                MAX_COUNTER_VALUE,
                self._bytes_read + _safe_count(snapshot.bytes_read),
            )
            self._bytes_written = min(
                MAX_COUNTER_VALUE,
                self._bytes_written + _safe_count(snapshot.bytes_written),
            )
            if safe_packet_id:
                self._merged_packets.add(safe_packet_id)
        return True

    def summary_line(self) -> str:
        snapshot = self.snapshot()
        values = " ".join(
            f"{name}={snapshot.counts.get(name, 0)}" for name in _OUTCOMES
        )
        return f"proof reuse: {values}"


__all__ = [
    "MAX_REASON_CODES",
    "PROOF_REUSE_METRICS_INTERFACE",
    "PROOF_REUSE_METRICS_SCHEMA",
    "ProofReuseMetricsSnapshot",
    "ProofReuseOutcome",
    "ProofReuseSessionMetrics",
]
