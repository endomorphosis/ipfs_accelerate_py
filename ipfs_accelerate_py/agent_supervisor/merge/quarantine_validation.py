"""Strict data validation shared by retained workspace custody guards."""

from __future__ import annotations

import json
from typing import Any


class QuarantineDenied(RuntimeError):
    """A retained quarantine cannot authorize this operation."""


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise QuarantineDenied(reason)


def strict_json(raw: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "quarantine_duplicate_json_key")
            result[key] = value
        return result

    def constant(_):
        raise QuarantineDenied("quarantine_nonfinite_json")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError) as exc:
        raise QuarantineDenied("quarantine_json_invalid") from exc
