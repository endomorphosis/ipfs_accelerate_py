"""Fail-open TypeSafe rank in front of tactician and hammer.

Drops only high-confidence unusable retrieve IDs. Never replaces
DeterministicDoctorTactician or DeterministicDoctorHammer. No key → keep
every candidate and always spend hammer.
"""

from __future__ import annotations

from typing import Any, Sequence

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    HIGH_CONFIDENCE,
    typesafe_permitted,
)

UNUSABLE_SCORE = 0.5


def _candidate_id(item: Any) -> str:
    if isinstance(item, dict):
        cand = item.get("candidate") if isinstance(item.get("candidate"), dict) else item
        return str(
            cand.get("candidate_ref") or cand.get("id") or item.get("id") or ""
        ).strip()
    cand = getattr(item, "candidate", item)
    return str(
        getattr(cand, "candidate_ref", "")
        or getattr(cand, "id", "")
        or getattr(item, "id", "")
        or ""
    ).strip()


def _candidate_summary(item: Any) -> str:
    if isinstance(item, dict):
        cand = item.get("candidate") if isinstance(item.get("candidate"), dict) else item
        return str(cand.get("path") or cand.get("symbol_id") or cand.get("kind") or "")[:240]
    cand = getattr(item, "candidate", item)
    return str(
        getattr(cand, "path", "")
        or getattr(cand, "symbol_id", "")
        or getattr(cand, "kind", "")
        or ""
    )[:240]


def select_retrieve_ids_for_tactician(
    candidates: Sequence[Any],
    *,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> tuple[str, ...]:
    """Keep allowlisted retrieve IDs. Drop unusable only at high confidence."""

    ordered = tuple(_candidate_id(item) for item in candidates if _candidate_id(item))
    if not ordered:
        return ()
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return ordered
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
        score_synthesis_candidate,
    )

    kept: list[str] = []
    dropped = 0
    by_id = {_candidate_id(item): item for item in candidates if _candidate_id(item)}
    for ident in ordered:
        receipt = score_synthesis_candidate(
            candidate_id=ident,
            allowlisted_ids=ordered,
            state={"summary": _candidate_summary(by_id.get(ident))},
            privacy_class=privacy_class,
            remote_disclosure_permitted=remote_disclosure_permitted,
            timeout=timeout,
        )
        unusable = (
            receipt.action == "scored"
            and float(receipt.score) < UNUSABLE_SCORE
            and float(receipt.confidence) >= HIGH_CONFIDENCE
        )
        if unusable:
            dropped += 1
            continue
        kept.append(ident)
    if not kept or dropped == 0:
        return ordered
    return tuple(kept)


def filter_candidates_for_tactician(
    candidates: Sequence[Any],
    *,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> tuple[Any, ...]:
    """Filter a candidate sequence. Fail-open to the original tuple."""

    items = tuple(candidates)
    try:
        kept_ids = set(
            select_retrieve_ids_for_tactician(
                items,
                privacy_class=privacy_class,
                remote_disclosure_permitted=remote_disclosure_permitted,
                timeout=timeout,
            )
        )
    except Exception:
        return items
    if not kept_ids:
        return items
    filtered = tuple(item for item in items if _candidate_id(item) in kept_ids)
    if not filtered:
        return items
    return filtered


__all__ = [
    "filter_candidates_for_tactician",
    "select_retrieve_ids_for_tactician",
]
