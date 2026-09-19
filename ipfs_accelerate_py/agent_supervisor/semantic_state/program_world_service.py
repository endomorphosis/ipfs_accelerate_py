"""SAWM-039 typed program-world service.

Deterministic JSON operations. Never writes DuckDB or completes tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ProgramWorldServiceError(ValueError):
    """Closed program-world service contract violation."""


@dataclass(frozen=True, slots=True)
class SemanticWorldStatusResult:
    status: str
    mode: str
    completion_authority: bool = False


@dataclass(frozen=True, slots=True)
class SemanticWorldResolveResult:
    query: str
    resolved: bool
    reason_code: str
    completion_authority: bool = False


class ProgramWorldService:
    INTERFACE = "SemanticWorldService@1"

    def status(self) -> SemanticWorldStatusResult:
        return SemanticWorldStatusResult(status="ready", mode="required")

    def resolve(self, query: Mapping[str, Any]) -> SemanticWorldResolveResult:
        q = str(query.get("query") or "")
        if not q:
            return SemanticWorldResolveResult(
                query=q, resolved=False, reason_code="empty_query"
            )
        return SemanticWorldResolveResult(
            query=q, resolved=True, reason_code="proposal_only"
        )

    def operation(self, name: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        allowed = {
            "world",
            "graph",
            "state",
            "trace",
            "transition",
            "call-target",
            "repair",
            "relation",
            "projection",
            "reuse",
            "procedure",
            "dogfood",
            "index",
            "benchmark",
        }
        if name not in allowed:
            raise ProgramWorldServiceError(f"unknown operation {name}")
        return {
            "operation": name,
            "payload": dict(payload or {}),
            "proposal_only": True,
            "completion_authority": False,
        }


def describe_controls() -> dict[str, Any]:
    return {
        "interface": ProgramWorldService.INTERFACE,
        "commands": ["status", "resolve", "world", "graph", "state", "trace", "reuse"],
        "completion_authority": False,
        "authoritative": False,
    }
