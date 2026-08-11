"""Shared implementation hard-timeout policy.

The implementation daemon enforces each task's hard cap while the bundle
supervisor sizes the parent watchdog around the largest cap in an execution
slice.  Both layers must resolve the same value or the parent can recycle a
healthy child before the child reaches its authorized deadline.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


DEFAULT_IMPLEMENTATION_TIMEOUT_SECONDS = 1800.0
DEFAULT_PROVIDER_IMPLEMENTATION_TIMEOUT_MULTIPLIER = 4.0

_TRUE_METADATA_VALUES = frozenset({"1", "true", "yes", "required"})


@dataclass(frozen=True)
class EffectiveImplementationHardTimeout:
    """The resolved hard cap and the policy source that authorized it."""

    seconds: float
    source: str


def _normalized_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, tuple[str, Any]]:
    normalized: dict[str, tuple[str, Any]] = {}
    for key, value in metadata.items():
        original = str(key).strip().lower()
        canonical = original.replace("_", " ")
        normalized.setdefault(canonical, (original, value))
    return normalized


def implementation_timeout_metadata_value(
    metadata: Mapping[str, Any],
    *keys: str,
    task_id: str = "",
) -> float | None:
    """Read one finite positive timeout from normalized task metadata."""

    normalized = _normalized_metadata(metadata)
    for key in keys:
        canonical = key.strip().lower().replace("_", " ")
        entry = normalized.get(canonical)
        if entry is None:
            continue
        field_name, raw_value = entry
        if raw_value in (None, ""):
            continue
        prefix = f"{task_id}: " if task_id else ""
        try:
            value = float(str(raw_value).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{prefix}{field_name} must be a finite positive number"
            ) from exc
        if (
            isinstance(raw_value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"{prefix}{field_name} must be a finite positive number"
            )
        return value
    return None


def effective_implementation_hard_timeout(
    metadata: Mapping[str, Any],
    *,
    configured_timeout: float,
    task_id: str = "",
) -> EffectiveImplementationHardTimeout:
    """Resolve one task's hard cap with daemon-authoritative precedence.

    Precedence is:

    1. ``implementation max timeout seconds``;
    2. ``implementation timeout seconds``;
    3. the provider-task multiplier; then
    4. the configured ordinary-task timeout.
    """

    prefix = f"{task_id}: " if task_id else ""
    if (
        isinstance(configured_timeout, bool)
        or not isinstance(configured_timeout, (int, float))
        or not math.isfinite(float(configured_timeout))
        or float(configured_timeout) <= 0
    ):
        raise ValueError(
            f"{prefix}implementation_timeout must be finite and positive"
        )
    configured = float(configured_timeout)
    explicit_max = implementation_timeout_metadata_value(
        metadata,
        "implementation max timeout seconds",
        "implementation maximum timeout seconds",
        task_id=task_id,
    )
    if explicit_max is not None:
        return EffectiveImplementationHardTimeout(
            seconds=explicit_max,
            source="task_metadata",
        )
    task_timeout = implementation_timeout_metadata_value(
        metadata,
        "implementation timeout seconds",
        "implementation timeout",
        task_id=task_id,
    )
    if task_timeout is not None:
        return EffectiveImplementationHardTimeout(
            seconds=task_timeout,
            source="task_metadata",
        )

    normalized = _normalized_metadata(metadata)
    requires_provider = str(
        normalized.get("requires provider", ("", ""))[1]
    ).strip().lower() in _TRUE_METADATA_VALUES
    if requires_provider:
        return EffectiveImplementationHardTimeout(
            seconds=(
                configured
                * DEFAULT_PROVIDER_IMPLEMENTATION_TIMEOUT_MULTIPLIER
            ),
            source="provider_task_progress",
        )
    return EffectiveImplementationHardTimeout(
        seconds=configured,
        source="configured_absolute",
    )
