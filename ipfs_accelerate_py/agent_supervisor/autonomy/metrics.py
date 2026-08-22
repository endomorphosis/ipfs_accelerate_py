"""Compact, content-bound operational metrics for the autonomy runtime.

``AutonomyMetrics@1`` records admitted model actions, durable writes, graph
scans, and budget refills.  Idle-cycle and safety-timer counters are
operational: they must not force an unchanged-state checkpoint write.  This
module never calls a model, opens a store, refills a budget, or authorizes
an effect.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..proof.formal_verification_contracts import canonical_json, content_identity
from .contracts import MAX_CANONICAL_RECORD_BYTES, MAX_IDENTIFIER_BYTES, MetaAction

AUTONOMY_METRICS_INTERFACE: Final[str] = "AutonomyMetrics@1"
AUTONOMY_METRICS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/metrics@1"
)
AUTONOMY_METRICS_DURABLE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/autonomy/metrics-durable@1"
)
MAX_AUTONOMY_METRICS_BYTES: Final[int] = MAX_CANONICAL_RECORD_BYTES
MAX_WAKE_KIND_COUNT: Final[int] = 64
_MAX_COUNTER: Final[int] = (1 << 63) - 1

_MODEL_ACTIONS = frozenset(
    {
        MetaAction.CALL_LOCAL_SMALL_MODEL,
        MetaAction.CALL_REMOTE_STANDARD_MODEL,
        MetaAction.CALL_REMOTE_STRONG_MODEL,
    }
)


class AutonomyMetricsError(ValueError):
    """Raised when a metrics snapshot is malformed or unbounded."""


def _counter(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AutonomyMetricsError(f"{name} must be a non-negative integer")
    if value < 0 or value > _MAX_COUNTER:
        raise AutonomyMetricsError(f"{name} is out of bounds")
    return value


def _kind(value: Any) -> str:
    text = str(getattr(value, "value", value) or "").strip()
    if (
        not text
        or len(text.encode("utf-8")) > MAX_IDENTIFIER_BYTES
        or any(char.isspace() for char in text)
        or "\x00" in text
    ):
        raise AutonomyMetricsError("wake kind must be a compact bounded identifier")
    return text


def _wake_counts(value: Any) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AutonomyMetricsError("wake_counts must be a mapping")
    if len(value) > MAX_WAKE_KIND_COUNT:
        raise AutonomyMetricsError("wake_counts exceeds the kind bound")
    result: dict[str, int] = {}
    for raw_kind, raw_count in value.items():
        kind = _kind(raw_kind)
        result[kind] = _counter(raw_count, "wake_counts")
    return dict(sorted(result.items()))


class AutonomyMetrics:
    """In-process counters whose durable projection is content-addressed.

    ``idle_cycles`` and ``safety_timer_wakes`` are excluded from
    :meth:`durable_identity` so a completed board's observation window cannot
    rewrite a checkpoint.
    """

    INTERFACE: ClassVar[str] = AUTONOMY_METRICS_INTERFACE
    SCHEMA: ClassVar[str] = AUTONOMY_METRICS_SCHEMA

    def __init__(self, snapshot: Mapping[str, Any] | None = None) -> None:
        self._model_calls = 0
        self._strong_model_calls = 0
        self._writes = 0
        self._scans = 0
        self._refills = 0
        self._admitted_actions = 0
        self._blocked_cycles = 0
        self._exhausted_cycles = 0
        self._cancelled_cycles = 0
        self._unavailable_cycles = 0
        self._idle_cycles = 0
        self._safety_timer_wakes = 0
        self._wake_counts: dict[str, int] = {}
        self._last_status = "idle"
        self._last_reason_codes: tuple[str, ...] = ()
        if snapshot is not None:
            self._restore(snapshot)

    def _restore(self, snapshot: Mapping[str, Any]) -> None:
        if not isinstance(snapshot, Mapping):
            raise AutonomyMetricsError("metrics snapshot must be a mapping")
        payload = dict(snapshot)
        payload.pop("metrics_id", None)
        schema = payload.get("schema", AUTONOMY_METRICS_SCHEMA)
        if schema not in {AUTONOMY_METRICS_SCHEMA, AUTONOMY_METRICS_DURABLE_SCHEMA}:
            raise AutonomyMetricsError("metrics snapshot schema mismatch")
        self._model_calls = _counter(payload.get("model_calls", 0), "model_calls")
        self._strong_model_calls = _counter(
            payload.get("strong_model_calls", 0), "strong_model_calls"
        )
        self._writes = _counter(payload.get("writes", 0), "writes")
        self._scans = _counter(payload.get("scans", 0), "scans")
        self._refills = _counter(payload.get("refills", 0), "refills")
        self._admitted_actions = _counter(
            payload.get("admitted_actions", 0), "admitted_actions"
        )
        self._blocked_cycles = _counter(payload.get("blocked_cycles", 0), "blocked_cycles")
        self._exhausted_cycles = _counter(
            payload.get("exhausted_cycles", 0), "exhausted_cycles"
        )
        self._cancelled_cycles = _counter(
            payload.get("cancelled_cycles", 0), "cancelled_cycles"
        )
        self._unavailable_cycles = _counter(
            payload.get("unavailable_cycles", 0), "unavailable_cycles"
        )
        self._idle_cycles = _counter(payload.get("idle_cycles", 0), "idle_cycles")
        self._safety_timer_wakes = _counter(
            payload.get("safety_timer_wakes", 0), "safety_timer_wakes"
        )
        self._wake_counts = _wake_counts(payload.get("wake_counts"))
        status = str(payload.get("last_status") or "idle").strip()
        if not status or len(status.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
            raise AutonomyMetricsError("last_status is unbounded")
        self._last_status = status
        reasons = payload.get("last_reason_codes") or ()
        if isinstance(reasons, str) or not isinstance(reasons, (list, tuple)):
            raise AutonomyMetricsError("last_reason_codes must be a string sequence")
        if len(reasons) > 64:
            raise AutonomyMetricsError("last_reason_codes exceeds the sequence bound")
        self._last_reason_codes = tuple(str(item) for item in reasons)
        if self._strong_model_calls > self._model_calls:
            raise AutonomyMetricsError("strong-model calls cannot exceed total model calls")

    @property
    def model_calls(self) -> int:
        return self._model_calls

    @property
    def strong_model_calls(self) -> int:
        return self._strong_model_calls

    @property
    def writes(self) -> int:
        return self._writes

    @property
    def scans(self) -> int:
        return self._scans

    @property
    def refills(self) -> int:
        return self._refills

    @property
    def admitted_actions(self) -> int:
        return self._admitted_actions

    @property
    def blocked_cycles(self) -> int:
        return self._blocked_cycles

    @property
    def exhausted_cycles(self) -> int:
        return self._exhausted_cycles

    @property
    def cancelled_cycles(self) -> int:
        return self._cancelled_cycles

    @property
    def unavailable_cycles(self) -> int:
        return self._unavailable_cycles

    @property
    def idle_cycles(self) -> int:
        return self._idle_cycles

    @property
    def safety_timer_wakes(self) -> int:
        return self._safety_timer_wakes

    @property
    def wake_counts(self) -> Mapping[str, int]:
        return MappingProxyType(dict(self._wake_counts))

    @property
    def last_status(self) -> str:
        return self._last_status

    @property
    def last_reason_codes(self) -> tuple[str, ...]:
        return self._last_reason_codes

    @property
    def unchanged_complete_idle(self) -> bool:
        """Whether a completed board has incurred no work side effects."""

        return (
            self._model_calls == 0
            and self._writes == 0
            and self._scans == 0
            and self._refills == 0
            and self._admitted_actions == 0
        )

    def record_wake(self, kind: Any, *, safety_timer: bool = False) -> None:
        key = _kind(kind)
        self._wake_counts[key] = _counter(
            self._wake_counts.get(key, 0) + 1, "wake_counts"
        )
        if safety_timer or key in {"window", "observation_window"}:
            self._safety_timer_wakes = _counter(
                self._safety_timer_wakes + 1, "safety_timer_wakes"
            )

    def record_idle(self, *, status: str = "idle", reason_codes: tuple[str, ...] = ()) -> None:
        self._idle_cycles = _counter(self._idle_cycles + 1, "idle_cycles")
        self._last_status = str(status or "idle")
        self._last_reason_codes = tuple(reason_codes)

    def record_scan(self) -> None:
        self._scans = _counter(self._scans + 1, "scans")

    def record_write(self) -> None:
        self._writes = _counter(self._writes + 1, "writes")

    def record_refill(self) -> None:
        """Explicit hole: the event runtime must never call this."""

        self._refills = _counter(self._refills + 1, "refills")

    def record_model_action(self, action: MetaAction | str | None) -> None:
        if action is None:
            return
        kind = action if isinstance(action, MetaAction) else MetaAction(str(action))
        if kind not in _MODEL_ACTIONS:
            return
        self._model_calls = _counter(self._model_calls + 1, "model_calls")
        if kind is MetaAction.CALL_REMOTE_STRONG_MODEL:
            self._strong_model_calls = _counter(
                self._strong_model_calls + 1, "strong_model_calls"
            )

    def record_status(self, status: str, *, reason_codes: tuple[str, ...] = ()) -> None:
        previous = self._last_status
        self._last_status = str(status or "")
        if not self._last_status or len(self._last_status.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
            raise AutonomyMetricsError("status must be a bounded identifier")
        self._last_reason_codes = tuple(reason_codes)
        if status in {"action_admitted", "progressing"}:
            self._admitted_actions = _counter(
                self._admitted_actions + 1, "admitted_actions"
            )
            return
        if status == previous:
            if status == "idle":
                self._idle_cycles = _counter(self._idle_cycles + 1, "idle_cycles")
            return
        if status == "blocked":
            self._blocked_cycles = _counter(self._blocked_cycles + 1, "blocked_cycles")
        elif status in {"budget_exhausted", "exhausted"}:
            self._exhausted_cycles = _counter(
                self._exhausted_cycles + 1, "exhausted_cycles"
            )
        elif status == "cancelled":
            self._cancelled_cycles = _counter(
                self._cancelled_cycles + 1, "cancelled_cycles"
            )
        elif status == "unavailable":
            self._unavailable_cycles = _counter(
                self._unavailable_cycles + 1, "unavailable_cycles"
            )
        elif status == "idle":
            self._idle_cycles = _counter(self._idle_cycles + 1, "idle_cycles")

    def _body(self, *, durable: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": AUTONOMY_METRICS_DURABLE_SCHEMA if durable else AUTONOMY_METRICS_SCHEMA,
            "interface": AUTONOMY_METRICS_INTERFACE,
            "model_calls": self._model_calls,
            "strong_model_calls": self._strong_model_calls,
            "refills": self._refills,
            "admitted_actions": self._admitted_actions,
            "blocked_cycles": self._blocked_cycles,
            "exhausted_cycles": self._exhausted_cycles,
            "cancelled_cycles": self._cancelled_cycles,
            "unavailable_cycles": self._unavailable_cycles,
        }
        if durable:
            return payload
        payload["writes"] = self._writes
        payload["scans"] = self._scans
        payload["wake_counts"] = dict(self._wake_counts)
        payload["idle_cycles"] = self._idle_cycles
        payload["safety_timer_wakes"] = self._safety_timer_wakes
        payload["last_status"] = self._last_status
        payload["last_reason_codes"] = list(self._last_reason_codes)
        return payload

    def durable_snapshot(self) -> Mapping[str, Any]:
        payload = self._body(durable=True)
        payload["metrics_id"] = content_identity(payload)
        encoded = canonical_json(payload).encode("utf-8")
        if len(encoded) > MAX_AUTONOMY_METRICS_BYTES:
            raise AutonomyMetricsError("metrics snapshot exceeds its bounded size")
        return MappingProxyType(payload)

    def durable_identity(self) -> str:
        return str(self.durable_snapshot()["metrics_id"])

    def snapshot(self) -> Mapping[str, Any]:
        payload = self._body(durable=False)
        payload["metrics_id"] = content_identity(payload)
        encoded = canonical_json(payload).encode("utf-8")
        if len(encoded) > MAX_AUTONOMY_METRICS_BYTES:
            raise AutonomyMetricsError("metrics snapshot exceeds its bounded size")
        return MappingProxyType(payload)

    def snapshot_json(self) -> str:
        return canonical_json(dict(self.snapshot()))

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any] | str) -> AutonomyMetrics:
        if isinstance(snapshot, str):
            try:
                payload = json.loads(snapshot)
            except json.JSONDecodeError as exc:
                raise AutonomyMetricsError("metrics snapshot is malformed") from exc
            if not isinstance(payload, Mapping):
                raise AutonomyMetricsError("metrics snapshot must contain an object")
            return cls(payload)
        return cls(snapshot)


__all__ = [
    "AUTONOMY_METRICS_DURABLE_SCHEMA",
    "AUTONOMY_METRICS_INTERFACE",
    "AUTONOMY_METRICS_SCHEMA",
    "MAX_AUTONOMY_METRICS_BYTES",
    "AutonomyMetrics",
    "AutonomyMetricsError",
]
