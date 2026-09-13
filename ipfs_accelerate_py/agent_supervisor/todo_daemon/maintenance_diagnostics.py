"""Bounded diagnostic projection; never maintenance or callback authority."""
from __future__ import annotations

import re
from typing import Any, Mapping

_SENSITIVE = re.compile(
    r"(?i)\b(?:bearer|authorization|cookie|password|passphrase|secret|"
    r"(?:api|access|auth|refresh|identity)[_ -]?(?:key|token))\b|"
    r"\btoken\s*[:=]|-----BEGIN[^\r\n]*PRIVATE KEY|\b[a-z][a-z0-9+.-]*://[^\s/@]+:[^\s/@]*@"
)


def _text(value: Any, limit: int) -> str:
    if type(value) is not str:
        return ""
    # Work and output are bounded, even for a malformed external projection.
    sample = value[:4096]
    if _SENSITIVE.search(sample):
        return "[redacted sensitive diagnostic]"
    sample = " ".join("".join(c if c.isprintable() else " " for c in sample).split())
    encoded = sample.encode("utf-8", errors="replace")
    if len(value) > 4096 or len(encoded) > limit:
        return encoded[:limit - 3].decode("utf-8", errors="ignore") + "..."
    return sample


def prelaunch_maintenance_observation(
    result: Mapping[str, Any], *, delay_seconds: float,
) -> dict[str, Any]:
    """Copy only diagnostic fields, preserving the caller's result untouched."""
    observation: dict[str, Any] = {
        "reason": _text(result.get("reason") or result.get("error"), 400) or "maintenance_blocked",
        "delay_seconds": delay_seconds,
    }
    projection = result.get("database_portal_reload_projection")
    source = "database_portal_reload_projection"
    if not isinstance(projection, Mapping):
        projection, source = result, "maintenance_result"
    if not any(type(projection.get(key)) is str and projection[key]
               for key in ("error_type", "error")):
        return observation
    error_type = projection.get("error_type")
    observation["maintenance_cause"] = {
        "diagnostic_only": True,
        "source": source,
        "reason": _text(projection.get("reason"), 160),
        "error_type": error_type if type(error_type) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", error_type) else "",
        "error": _text(projection.get("error"), 400),
    }
    return observation
