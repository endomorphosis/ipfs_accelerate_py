"""Advisory TypeSafe context rerank and admissibility lints.

Does not compile capsules, does not write patches, and does not drop
kernel-proved candidates. Unknown IDs are ignored. No key → original
snippet order and no lint flags.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.typesafe_inference import Noul, Score
from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    AdvisoryReceipt,
    typesafe_permitted,
)

MAX_SNIPPETS = 8
MAX_SNIPPET_CHARS = 400


def _snippet_id(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or row.get("reference_id") or "").strip()


def _snippet_text(row: Mapping[str, Any]) -> str:
    text = str(row.get("text") or row.get("excerpt") or row.get("summary") or "")
    return text[:MAX_SNIPPET_CHARS]


def rerank_questions() -> dict[str, Any]:
    return {
        "needed": Noul(
            instructions={
                "question": "Is `snippet.text` needed to decide `obligation.id`?",
                "inspect": "`snippet.text`",
            },
        ),
        "relevance": Score(
            instructions={
                "question": "How relevant is `snippet.text` to `obligation.id`?",
                "inspect": "`snippet.text`",
            },
            criteria=["off-topic", "supporting", "decisive"],
        ),
    }


def compose_snippet_score(result: Any) -> float:
    nouls = getattr(result, "nouls", None) or {}
    scores = getattr(result, "scores", None) or {}
    needed = float(getattr(nouls.get("needed"), "noul", 0.0) or 0.0)
    relevance = float(getattr(scores.get("relevance"), "score", 0.0) or 0.0)
    return max(0.0, min(1.0, 0.4 * needed + 0.6 * (relevance / 2.0)))


def rerank_allowlisted_snippets(
    snippets: Sequence[Mapping[str, Any]],
    *,
    obligation_id: str,
    allowlisted_ids: Sequence[str],
    top_k: int = MAX_SNIPPETS,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> tuple[str, ...]:
    """Return allowlisted snippet IDs, TypeSafe-ordered. Fail-open to input order."""

    allowed = {str(item).strip() for item in allowlisted_ids if str(item).strip()}
    ordered = []
    seen: set[str] = set()
    for row in snippets:
        ident = _snippet_id(row)
        if not ident or ident not in allowed or ident in seen:
            continue
        seen.add(ident)
        ordered.append(row)
    keep = tuple(_snippet_id(row) for row in ordered[: max(1, int(top_k))])
    if not keep:
        return ()
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return keep
    from ipfs_accelerate_py.typesafe_inference import system_one

    scored: list[tuple[float, str]] = []
    for row in ordered[: max(1, int(top_k))]:
        ident = _snippet_id(row)
        state = {
            "obligation": {"id": str(obligation_id or "")[:128]},
            "snippet": {
                "id": ident,
                "path": str(row.get("path") or "")[:128],
                "text": _snippet_text(row),
            },
        }
        try:
            result = system_one(state, rerank_questions(), timeout=timeout)
        except Exception:
            return keep
        scored.append((compose_snippet_score(result), ident))
    scored.sort(reverse=True)
    return tuple(ident for _score, ident in scored)


def _item_id(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("reference_id") or item.get("id") or "").strip()
    return str(getattr(item, "reference_id", "") or getattr(item, "id", "") or "").strip()


def _item_required(item: Any) -> bool:
    if isinstance(item, Mapping):
        return bool(item.get("required"))
    return bool(getattr(item, "required", False))


def _item_text(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("summary") or item.get("text") or "")[:MAX_SNIPPET_CHARS]
    return str(getattr(item, "summary", "") or getattr(item, "text", "") or "")[:MAX_SNIPPET_CHARS]


def prepare_evidence_for_compile(
    evidence: Sequence[Any],
    *,
    obligation_id: str,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> tuple[Any, ...]:
    """Keep required items first; rerank optional allowlisted IDs. Fail-open."""

    items = tuple(evidence)
    required = [item for item in items if _item_required(item)]
    optional = [item for item in items if not _item_required(item)]
    if not optional:
        return items
    snippets = [
        {"id": _item_id(item), "text": _item_text(item), "path": str(getattr(item, "path", "") or "")}
        for item in optional
        if _item_id(item)
    ]
    allowlisted = tuple(_item_id(item) for item in optional if _item_id(item))
    try:
        order = rerank_allowlisted_snippets(
            snippets,
            obligation_id=obligation_id,
            allowlisted_ids=allowlisted,
            privacy_class=privacy_class,
            remote_disclosure_permitted=remote_disclosure_permitted,
            timeout=timeout,
        )
    except Exception:
        return items
    by_id = {_item_id(item): item for item in optional if _item_id(item)}
    reranked = [by_id[ident] for ident in order if ident in by_id]
    leftover = [item for item in optional if _item_id(item) not in set(order)]
    return tuple((*required, *reranked, *leftover))


def lint_questions() -> dict[str, Any]:
    return {
        "matches_obligation": Noul(
            instructions={
                "question": "Does `patch.summary` address `obligation.text`?",
                "compare": ["`patch.summary`", "`obligation.text`"],
            },
        ),
        "claims_kernel_without_receipt": Noul(
            instructions={
                "question": "Does `patch.summary` claim KERNEL_VERIFIED without `patch.has_kernel_receipt`?",
                "inspect": "`patch`",
            },
        ),
        "review_needed": Score(
            instructions={
                "question": "How strongly should a human review this before write?",
            },
            criteria=["none", "optional", "required"],
        ),
    }


def lint_admissibility(
    *,
    obligation_id: str,
    obligation_text: str = "",
    patch_summary: str = "",
    claimed_kernel_verified: bool = False,
    has_kernel_receipt: bool = False,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> AdvisoryReceipt:
    """Advisory flags only. Never blocks a write or kernel-proved candidate."""

    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return AdvisoryReceipt(
            action="skipped",
            reason_codes=("privacy_or_unconfigured",),
        )
    from ipfs_accelerate_py.typesafe_inference import system_one

    state = {
        "obligation": {
            "id": str(obligation_id or "")[:128],
            "text": str(obligation_text or "")[:400],
        },
        "patch": {
            "summary": str(patch_summary or "")[:400],
            "claimed_kernel_verified": bool(claimed_kernel_verified),
            "has_kernel_receipt": bool(has_kernel_receipt),
        },
    }
    try:
        result = system_one(state, lint_questions(), timeout=timeout)
    except Exception:
        return AdvisoryReceipt(action="abstain", reason_codes=("typesafe_error",))
    nouls = getattr(result, "nouls", None) or {}
    scores = getattr(result, "scores", None) or {}
    matches = float(getattr(nouls.get("matches_obligation"), "noul", 0.0) or 0.0)
    false_claim = float(
        getattr(nouls.get("claims_kernel_without_receipt"), "noul", 0.0) or 0.0
    )
    review = float(getattr(scores.get("review_needed"), "score", 0.0) or 0.0)
    reasons = ["composed_in_code", "advisory_lint_only"]
    if matches < 0.4:
        reasons.append("obligation_mismatch")
    if false_claim >= 0.6 and not has_kernel_receipt:
        reasons.append("false_kernel_claim")
    if review >= 1.5:
        reasons.append("human_review_suggested")
    return AdvisoryReceipt(
        action="linted",
        noul=matches,
        score=review,
        reason_codes=tuple(reasons),
        accepted_as_authority=False,
    )


CLAIM_MARKERS: tuple[str, ...] = (
    "kernel_verified",
    "proved",
    "proof complete",
    "qed",
    "theorem holds",
    "all tests passed",
)


def extract_claim_spans(text: str) -> tuple[dict[str, str], ...]:
    """Split text into short claim spans that look like proof/test assertions."""

    blob = str(text or "")
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", blob) if part.strip()]
    spans: list[dict[str, str]] = []
    for index, part in enumerate(parts):
        lowered = part.casefold()
        if any(marker in lowered for marker in CLAIM_MARKERS):
            spans.append({"id": f"claim-{index}", "text": part[:240]})
        if len(spans) >= 8:
            break
    return tuple(spans)


def citation_questions() -> dict[str, Any]:
    return {
        "supported": Noul(
            instructions={
                "question": "Is `claim.text` supported by one of `receipt_ids`?",
                "compare": ["`claim.text`", "`receipt_ids`"],
                "focus": "KERNEL_VERIFIED or test claims need a matching receipt id.",
            },
        ),
    }


def cite_claim_spans(
    spans: Sequence[Mapping[str, Any]],
    *,
    receipt_ids: Sequence[str],
    allowlisted_ids: Sequence[str],
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> tuple[str, ...]:
    """Return allowlisted claim IDs that look unsupported. Never completes a task."""

    allowed = {str(item).strip() for item in allowlisted_ids if str(item).strip()}
    receipts = tuple(str(item).strip()[:128] for item in receipt_ids if str(item).strip())
    selected = [
        row
        for row in spans
        if isinstance(row, Mapping) and str(row.get("id") or "").strip() in allowed
    ]
    if not selected:
        return ()
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        return ()
    from ipfs_accelerate_py.typesafe_inference import system_one

    unsupported: list[str] = []
    for row in selected[:8]:
        ident = str(row.get("id") or "").strip()
        state = {
            "claim": {"id": ident, "text": str(row.get("text") or "")[:240]},
            "receipt_ids": list(receipts[:16]),
        }
        try:
            result = system_one(state, citation_questions(), timeout=timeout)
        except Exception:
            return ()
        noul = getattr((getattr(result, "nouls", None) or {}).get("supported"), "noul", 1.0)
        try:
            supported = float(noul or 0.0)
        except (TypeError, ValueError):
            supported = 1.0
        if supported < 0.4:
            unsupported.append(ident)
    return tuple(unsupported)


__all__ = [
    "cite_claim_spans",
    "citation_questions",
    "compose_snippet_score",
    "extract_claim_spans",
    "lint_admissibility",
    "lint_questions",
    "prepare_evidence_for_compile",
    "rerank_allowlisted_snippets",
    "rerank_questions",
]
