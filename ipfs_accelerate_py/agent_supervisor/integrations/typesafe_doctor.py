"""Fail-open TypeSafe rank in front of tactician and hammer.

Drops only high-confidence unusable retrieve IDs. Never replaces
DeterministicDoctorTactician or DeterministicDoctorHammer. No key → keep
every candidate and always spend hammer.
"""

from __future__ import annotations

import threading
from typing import Any, Mapping, Sequence

from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    HIGH_CONFIDENCE,
    typesafe_permitted,
)

UNUSABLE_SCORE = 0.5
_LAST_HAMMER_HINT = threading.local()


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
        score_synthesis_candidates_fanout,
    )

    kept: list[str] = []
    dropped = 0
    by_id = {_candidate_id(item): item for item in candidates if _candidate_id(item)}
    fanout: dict[str, Any] = {}
    try:
        fanout = score_synthesis_candidates_fanout(
            tuple(
                (ident, _candidate_summary(by_id.get(ident)))
                for ident in ordered[:8]
            ),
            allowlisted_ids=ordered,
            privacy_class=privacy_class,
            remote_disclosure_permitted=remote_disclosure_permitted,
            timeout=timeout,
        )
    except Exception:
        fanout = {}
    for ident in ordered:
        receipt = fanout.get(ident)
        if receipt is None:
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


def order_candidates_for_hammer(
    candidates: Sequence[Any],
    *,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> tuple[Any, ...]:
    """Reorder candidates by TypeSafe quality. Keep all. Fail-open original order."""

    items = tuple(candidates)
    if len(items) < 2:
        return items
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return items
    from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
        score_synthesis_candidate,
        score_synthesis_candidates_fanout,
    )

    ordered_ids = tuple(_candidate_id(item) for item in items if _candidate_id(item))
    if len(ordered_ids) < 2:
        return items
    fanout: dict[str, Any] = {}
    try:
        fanout = score_synthesis_candidates_fanout(
            tuple((_candidate_id(item), _candidate_summary(item)) for item in items),
            allowlisted_ids=ordered_ids,
            privacy_class=privacy_class,
            remote_disclosure_permitted=remote_disclosure_permitted,
            timeout=timeout,
        )
    except Exception:
        fanout = {}
    ranked: list[tuple[float, int, Any]] = []
    scores: dict[str, float] = {}
    for index, item in enumerate(items):
        ident = _candidate_id(item) or f"anon-{index}"
        try:
            receipt = fanout.get(ident)
            if receipt is None:
                receipt = score_synthesis_candidate(
                    candidate_id=ident,
                    allowlisted_ids=ordered_ids or (ident,),
                    state={"summary": _candidate_summary(item)},
                    privacy_class=privacy_class,
                    remote_disclosure_permitted=remote_disclosure_permitted,
                    timeout=timeout,
                )
            quality = float(receipt.score) if receipt.action == "scored" else 0.0
        except Exception:
            return items
        scores[ident] = quality
        ranked.append((-quality, index, item))
    ranked.sort()
    hint = {
        "typesafe_hint_only": True,
        "accepted_as_authority": False,
        "ordered_ids": [_candidate_id(item) or f"anon-{idx}" for _q, idx, item in ranked],
        "scores": scores,
    }
    _LAST_HAMMER_HINT.value = dict(hint)
    return tuple(item for _q, _i, item in ranked)


def last_hammer_hint() -> dict[str, Any]:
    value = getattr(_LAST_HAMMER_HINT, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def hammer_timeout_hint(
    *,
    finding_id: str,
    declaration: str = "",
    english: str = "",
    smtlib: str = "",
) -> dict[str, Any]:
    """Advisory sat/unsat after hammer timeout. Never KERNEL_VERIFIED."""

    empty: dict[str, Any] = {}
    if not typesafe_permitted():
        return empty
    try:
        from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
            triage_smt,
        )

        text = str(english or declaration or finding_id or "").strip()
        receipt = triage_smt(
            english=text,
            smtlib=str(smtlib or ""),
            case_id=str(finding_id or "hammer-timeout")[:64],
            complexity="hard",
        )
        hint = {
            "typesafe_hint_only": True,
            "claim_status": receipt.claim_status,
            "confidence": round(float(receipt.confidence or 0.0), 4),
            "accepted_as_authority": False,
        }
        try:
            from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_calibration import (
                record_sample,
            )
            from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
                is_trap_family,
            )

            record_sample(
                family=str(finding_id or "hammer-timeout")[:64],
                predicted=str(receipt.claim_status or ""),
                actual="timeout",
                confidence=float(receipt.confidence or 0.0),
                trap_family=is_trap_family(
                    smtlib=smtlib, case_id=str(finding_id or ""), complexity="hard"
                ),
                case_id=str(finding_id or ""),
            )
        except Exception:
            pass
        _LAST_HAMMER_HINT.value = dict(hint)
        return hint
    except Exception:
        return empty


__all__ = [
    "filter_candidates_for_tactician",
    "hammer_timeout_hint",
    "last_hammer_hint",
    "order_candidates_for_hammer",
    "select_retrieve_ids_for_tactician",
]
