"""SAWM-043 negative-memory invalidation explanation.

Explanations are diagnostic only. They cannot complete a board task or
publish a generation root by themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RequiredModeCapstoneReceipt:
    explained: bool
    generation_published: bool = False
    completion_authority: bool = False


@dataclass
class NegativeMemoryInvalidationExplainer:
    def explain_negative_memory_invalidation(
        self, record: Mapping[str, Any]
    ) -> dict[str, Any]:
        reason = str(record.get("reason_code") or "unspecified")
        mismatches = tuple(record.get("mismatches") or ())
        return {
            "reason_code": reason,
            "mismatches": list(mismatches),
            "explanation": f"reuse key invalidated because {reason}",
            "completion_authority": False,
            "generation_published": False,
        }


def explain_negative_memory_invalidation(record: Mapping[str, Any]) -> dict[str, Any]:
    return NegativeMemoryInvalidationExplainer().explain_negative_memory_invalidation(record)


def explain_negative_memory(
    record: Mapping[str, Any],
    *,
    tree_id: str | None = None,
    current_tree_id: str | None = None,
) -> dict[str, Any]:
    payload = explain_negative_memory_invalidation(record)
    if tree_id is not None and current_tree_id is not None and tree_id != current_tree_id:
        payload["reason_code"] = "stale_tree"
        payload["explanation"] = "reuse key invalidated because stale_tree"
    return payload
