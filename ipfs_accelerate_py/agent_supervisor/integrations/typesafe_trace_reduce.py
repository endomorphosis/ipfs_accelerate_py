"""Map-reduce TypeSafe classification over redacted supervisor traces.

Classifies stall vs kernel-wait vs provider-down vs fail. Counts live in
code. ``may_complete_task`` is always false. No key → deterministic labels
from status/error_code only. Events never include prompts or grant payloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.typesafe_inference import Choice, Noul
from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    typesafe_permitted,
)

TRACE_LABELS: tuple[str, ...] = (
    "success",
    "stall",
    "kernel_wait",
    "provider_down",
    "flaky_fail",
    "real_fail",
    "other",
)
MAX_EVENTS = 16


def redact_event(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "operation": str(row.get("operation") or row.get("kind") or "")[:64],
        "status": str(row.get("status") or "")[:32],
        "error_code": str(row.get("error_code") or row.get("reason") or "")[:64],
        "authority": str(row.get("authority") or "")[:32],
        "dry_run": bool(row.get("dry_run")),
    }


def classify_event_deterministic(redacted: Mapping[str, Any]) -> str:
    status = str(redacted.get("status") or "").casefold()
    err = str(redacted.get("error_code") or "").casefold()
    if status in {"succeeded", "ok", "success"}:
        return "success"
    if "timeout" in err or "stall" in err or "deadline" in err:
        return "stall"
    if "kernel" in err or "lean" in err or "smt" in err:
        return "kernel_wait"
    if any(token in err for token in ("unauthenticated", "429", "provider", "overloaded", "529")):
        return "provider_down"
    if "flaky" in err:
        return "flaky_fail"
    if status in {"failed", "error", "denied"}:
        return "real_fail"
    return "other"


def classification_questions() -> dict[str, Any]:
    return {
        "label": Choice(
            instructions={
                "question": "Which closed label describes this redacted supervisor event?",
                "inspect": "`event`",
            },
            criteria={
                "success": {"what": "The operation succeeded", "not_for": "A failed or stalled event"},
                "stall": {"what": "Timeout, deadline, or idle wait", "not_for": "A finished failure"},
                "kernel_wait": {"what": "Blocked on Lean/SMT/kernel", "not_for": "A provider HTTP error"},
                "provider_down": {
                    "what": "Model/CLI provider auth, rate limit, or overload",
                    "not_for": "A kernel or logic failure",
                },
                "flaky_fail": {"what": "Intermittent test or infra flake", "not_for": "A reproducible real fail"},
                "real_fail": {"what": "Reproducible task or check failure", "not_for": "Timeout or provider outage"},
                "other": {"what": "None of the listed labels", "not_for": "A clear success or fail"},
            },
        ),
        "retryable": Noul(
            instructions={
                "question": "Should code retry this event without completing the task?",
                "inspect": "`event.status`",
            },
        ),
    }


@dataclass(frozen=True)
class TraceReduceReport:
    labels: tuple[str, ...]
    counts: Mapping[str, int]
    retryable: tuple[bool, ...] = ()
    source: str = "deterministic"
    may_complete_task: bool = False
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "may_complete_task", False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "counts": dict(self.counts),
            "retryable": list(self.retryable),
            "source": self.source,
            "may_complete_task": False,
            "reason_codes": list(self.reason_codes),
        }


def _counts(labels: Sequence[str]) -> dict[str, int]:
    tallies = {label: 0 for label in TRACE_LABELS}
    for label in labels:
        key = label if label in tallies else "other"
        tallies[key] += 1
    return tallies


def reduce_events(
    events: Sequence[Mapping[str, Any]],
    *,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> TraceReduceReport:
    """Classify a bounded event list. Never marks a task complete."""

    rows = [redact_event(item) for item in events[:MAX_EVENTS] if isinstance(item, Mapping)]
    if not rows:
        return TraceReduceReport(labels=(), counts=_counts(()), source="empty")
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        labels = tuple(classify_event_deterministic(item) for item in rows)
        return TraceReduceReport(
            labels=labels,
            counts=_counts(labels),
            source="deterministic",
            reason_codes=("privacy_or_unconfigured",),
        )
    from ipfs_accelerate_py.typesafe_inference import system_one

    labels: list[str] = []
    retryable: list[bool] = []
    source = "typesafe"
    for item in rows:
        try:
            result = system_one({"event": item}, classification_questions(), timeout=timeout)
        except Exception:
            source = "deterministic"
            labels = [classify_event_deterministic(row) for row in rows]
            return TraceReduceReport(
                labels=tuple(labels),
                counts=_counts(labels),
                source=source,
                reason_codes=("typesafe_error_fail_open",),
            )
        choice = result.choices.get("label")
        nominated = str(getattr(choice, "choice", "") or "")
        if nominated not in TRACE_LABELS:
            nominated = classify_event_deterministic(item)
        labels.append(nominated)
        noul = result.nouls.get("retryable")
        retryable.append(float(getattr(noul, "noul", 0.0) or 0.0) >= 0.5)
    return TraceReduceReport(
        labels=tuple(labels),
        counts=_counts(labels),
        retryable=tuple(retryable),
        source=source,
        reason_codes=("composed_in_code",),
    )


def reduce_jsonl(
    path: str | Path,
    *,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> TraceReduceReport:
    """Read a JSONL audit/board file and classify a bounded prefix."""

    events: list[Mapping[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return TraceReduceReport(
            labels=(),
            counts=_counts(()),
            source="empty",
            reason_codes=("unreadable",),
        )
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            events.append(payload)
        if len(events) >= MAX_EVENTS:
            break
    return reduce_events(
        events,
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
        timeout=timeout,
    )


__all__ = [
    "TRACE_LABELS",
    "TraceReduceReport",
    "classify_event_deterministic",
    "reduce_events",
    "reduce_jsonl",
    "redact_event",
]
