"""Advisory semantic task-kind routing beside the Intelligence Index.

TypeSafe may escalate the closed kind (higher intelligence floor). It may
downgrade only at high confidence. No key → board kind unchanged. Never
selects the ``typesafe`` generate_text provider.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ipfs_accelerate_py.typesafe_inference import Choice, Noul, Score
from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    HIGH_CONFIDENCE,
    typesafe_permitted,
)

ROUTABLE_KINDS: tuple[str, ...] = (
    "inventory",
    "legal",
    "planning",
    "review",
    "implementation",
    "repair",
    "merge",
    "scientific",
    "agent",
    "standard",
)


def _redacted_task_state(task: Any) -> dict[str, Any]:
    if isinstance(task, str):
        return {"title": task[:240], "kind": "", "summary": ""}
    if not isinstance(task, Mapping):
        return {"title": "", "kind": "", "summary": ""}
    kind = str(task.get("kind") or task.get("task_kind") or "")
    metadata = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
    if not kind:
        kind = str(metadata.get("kind") or metadata.get("task_kind") or "")
    return {
        "kind": kind[:64],
        "title": str(task.get("title") or "")[:240],
        "summary": str(task.get("summary") or metadata.get("summary") or "")[:480],
        "objective_id": str(task.get("objective_id") or "")[:128],
    }


def routing_questions() -> dict[str, Any]:
    return {
        "intent": Choice(
            instructions={
                "question": "Which closed supervisor task kind best matches this item?",
                "inspect": "`title`",
                "focus": "Use only the listed kinds. Do not invent a kind.",
            },
            criteria={
                item: {
                    "what": item,
                    "not_for": "any other listed kind",
                }
                for item in ROUTABLE_KINDS
            },
        ),
        "is_coding_task": Noul(
            instructions={
                "question": "Is this a coding, repair, or implementation task?",
                "inspect": "`title`",
            },
        ),
        "is_proof_bearing": Noul(
            instructions={
                "question": "Does this item require a proof, kernel, or SMT check?",
            },
        ),
        "is_side_effecting": Noul(
            instructions={
                "question": "Would acting on this item write files or change git state?",
            },
        ),
        "difficulty": Score(
            instructions={
                "question": "How hard is this item for a coding model?",
                "inspect": "`summary`",
            },
            criteria=["easy", "standard", "hard"],
        ),
    }


def compose_task_kind(
    result: Any,
    *,
    board_kind: str,
) -> tuple[str, float, tuple[str, ...]]:
    from ipfs_accelerate_py.llm_allocation.intelligence_index import (
        TASK_KIND_INTELLIGENCE,
        intelligence_floor_for_task,
    )

    choices = getattr(result, "choices", None) or {}
    intent = choices.get("intent") if isinstance(choices, Mapping) else None
    nominated = str(getattr(intent, "choice", "") or "").strip().casefold().replace("-", "_")
    confidence = float(getattr(intent, "confidence", 0.0) or 0.0)
    reasons = ["composed_in_code"]
    board = str(board_kind or "standard").strip().casefold().replace("-", "_")
    if board not in TASK_KIND_INTELLIGENCE:
        board = "standard"
    if nominated not in TASK_KIND_INTELLIGENCE:
        reasons.append("unknown_kind_keep_board")
        return board, confidence, tuple(reasons)
    board_floor = intelligence_floor_for_task(board)
    advised_floor = intelligence_floor_for_task(nominated)
    if advised_floor >= board_floor:
        reasons.append("escalate_or_same_floor")
        return nominated, confidence, tuple(reasons)
    if confidence >= HIGH_CONFIDENCE:
        reasons.append("confident_downgrade")
        return nominated, confidence, tuple(reasons)
    reasons.append("keep_board_kind")
    return board, confidence, tuple(reasons)


def advise_board_task_kind(
    task: Any,
    *,
    fallback: str = "",
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> str:
    """Return a closed kind. Fail-open to ``fallback`` / board kind."""

    from ipfs_accelerate_py.llm_allocation.intelligence_index import board_task_kind

    board = str(fallback or "").strip() or board_task_kind(task)
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return board
    from ipfs_accelerate_py.typesafe_inference import system_one

    try:
        result = system_one(
            _redacted_task_state(task),
            routing_questions(),
            timeout=timeout,
        )
    except Exception:
        return board
    nominated, _confidence, _reasons = compose_task_kind(result, board_kind=board)
    return nominated


__all__ = [
    "ROUTABLE_KINDS",
    "advise_board_task_kind",
    "compose_task_kind",
    "routing_questions",
]
