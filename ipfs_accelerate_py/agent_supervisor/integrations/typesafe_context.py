"""Advisory TypeSafe context rerank and admissibility lints.

Does not compile capsules, does not write patches, and does not drop
kernel-proved candidates. Unknown IDs are ignored. No key → original
snippet order and no lint flags.
"""

from __future__ import annotations

import re
import threading
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.typesafe_inference import Noul, Score, noul_yes_no
from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    AdvisoryReceipt,
    typesafe_permitted,
)

MAX_SNIPPETS = 8
MAX_SNIPPET_CHARS = 400
CITATION_AUTO_ACCEPT = 0.8
CITATION_CHOICES = ("supports", "contradicts", "says_nothing")
PICK_NONE = "none"
PICK_MAX = 8
DATE_REVIEW_BELOW = 0.60
DATE_YEAR_MIN = 1900
DATE_YEAR_MAX = 2050
FIND_MAX_LINES = 17
FIND_EXISTS_HIGH = 0.70
FIND_EXISTS_LOW = 0.35
RANK_SHORTLIST = 3
FITS_THRESHOLD = 0.30
JOIN_AFTER_DANGLING = 0.2
JOIN_AFTER_TERMINAL = 0.5
CLASSIFY_MAX_BLOCKS = 8
HEADING_MAX_CHARS = 90
STEP_THRESHOLD = 0.5
BLOCK_TYPES = (
    "heading",
    "paragraph",
    "list_item",
    "quote",
    "code",
    "callout",
)
HEADING_LEVELS = ("title", "section", "subsection")
CALLOUT_KINDS = ("note", "tip", "warning")
_TERMINAL_END = re.compile(r'[.!?:;…]["\')\]]*$')
DATE_MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
DATE_WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_YEAR_IN_TEXT = re.compile(r"\b(19\d{2}|20\d{2})\b")
_CURLY_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
_LAST_SOURCE_EDIT_LINT = threading.local()
_LAST_ARTIFACT_VIEW = threading.local()
_LAST_CITATION = threading.local()
_LAST_PICK = threading.local()
_LAST_DATE = threading.local()
_LAST_FIND = threading.local()
_LAST_STITCH = threading.local()


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


def _pairwise_keys(ids: Sequence[str]) -> tuple[tuple[str, str, str], ...]:
    items = [str(ident).strip() for ident in ids if str(ident).strip()]
    pairs: list[tuple[str, str, str]] = []
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            pairs.append((f"better_{left}_than_{right}", left, right))
    return tuple(pairs)


def pairwise_questions(
    ids: Sequence[str],
    *,
    left_path: str,
    right_path: str,
) -> dict[str, Any]:
    """One noul per unordered pair. Never invents ids."""

    questions: dict[str, Any] = {}
    for key, left, right in _pairwise_keys(ids):
        questions[key] = Noul(
            instructions={
                "question": (
                    f"Is `{left_path.format(id=left)}` more relevant than "
                    f"`{right_path.format(id=right)}`?"
                ),
                "compare": [
                    f"`{left_path.format(id=left)}`",
                    f"`{right_path.format(id=right)}`",
                ],
            },
        )
    return questions


def compose_pairwise_order(
    ids: Sequence[str],
    *,
    nouls: Mapping[str, Any] | None = None,
    independent: Mapping[str, float] | None = None,
) -> tuple[str, ...]:
    """Tournament order from pairwise nouls. Unknown ids stay in input order."""

    ordered = tuple(str(ident).strip() for ident in ids if str(ident).strip())
    if len(ordered) < 2:
        return ordered
    scores = {ident: float(independent.get(ident, 0.0) or 0.0) for ident in ordered} if independent else {ident: 0.0 for ident in ordered}
    wins = {ident: 0.0 for ident in ordered}
    blob = nouls or {}
    for key, left, right in _pairwise_keys(ordered):
        if left not in wins or right not in wins:
            continue
        raw = blob.get(key)
        if isinstance(raw, (int, float)):
            noul = float(raw)
        else:
            noul = float(getattr(raw, "noul", 0.5) or 0.5)
        wins[left] += noul
        wins[right] += 1.0 - noul
    return tuple(
        sorted(
            ordered,
            key=lambda ident: (
                -wins[ident],
                -scores.get(ident, 0.0),
                ordered.index(ident),
            ),
        )
    )


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

    rows = ordered[: max(1, int(top_k))]
    state = {
        "obligation": {"id": str(obligation_id or "")[:128]},
        "snippets": {
            _snippet_id(row): {
                "id": _snippet_id(row),
                "path": str(row.get("path") or "")[:128],
                "text": _snippet_text(row),
            }
            for row in rows
        },
    }
    questions: dict[str, Any] = {}
    for row in rows:
        ident = _snippet_id(row)
        questions[f"needed_{ident}"] = Noul(
            instructions={
                "question": (
                    f"Is `snippets.{ident}.text` needed to decide `obligation.id`?"
                ),
                "inspect": f"`snippets.{ident}.text`",
            },
        )
        questions[f"relevance_{ident}"] = Score(
            instructions={
                "question": (
                    f"How relevant is `snippets.{ident}.text` to `obligation.id`?"
                ),
                "inspect": f"`snippets.{ident}.text`",
            },
            criteria=["off-topic", "supporting", "decisive"],
        )
    questions.update(
        pairwise_questions(
            keep,
            left_path="snippets.{id}.text",
            right_path="snippets.{id}.text",
        )
    )
    try:
        result = system_one(state, questions, timeout=timeout)
    except Exception:
        return keep
    nouls = getattr(result, "nouls", None) or {}
    scores = getattr(result, "scores", None) or {}
    independent: dict[str, float] = {}
    for ident in keep:
        adapter = SimpleNamespace(
            nouls={"needed": nouls.get(f"needed_{ident}")},
            scores={"relevance": scores.get(f"relevance_{ident}")},
        )
        independent[ident] = compose_snippet_score(adapter)
    return compose_pairwise_order(keep, nouls=nouls, independent=independent)


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


def last_source_edit_lint() -> dict[str, Any]:
    value = getattr(_LAST_SOURCE_EDIT_LINT, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def observe_source_edit_lint(
    *,
    operator_id: str,
    relative_path: str,
    claimed_kernel_verified: bool = False,
    has_kernel_receipt: bool = False,
) -> Optional[AdvisoryReceipt]:
    """Advisory lint only. Never changes source-edit admission. Never raises."""

    try:
        receipt = lint_admissibility(
            obligation_id=str(operator_id or "")[:128],
            obligation_text=str(relative_path or "")[:400],
            patch_summary=f"{operator_id} {relative_path}"[:400],
            claimed_kernel_verified=claimed_kernel_verified,
            has_kernel_receipt=has_kernel_receipt,
        )
    except Exception:
        return None
    payload = receipt.to_dict()
    payload["accepted_as_authority"] = False
    _LAST_SOURCE_EDIT_LINT.value = payload
    return receipt


_LAST_STATIC_LINT = threading.local()


def last_static_lint() -> dict[str, Any]:
    value = getattr(_LAST_STATIC_LINT, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def lint_static_span(
    *,
    obligation_id: str,
    path: str = "",
    summary: str = "",
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> AdvisoryReceipt:
    """Semantic lint in front of RUN_LOCAL_STATIC_ANALYSIS. Analyzer still runs."""

    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        receipt = AdvisoryReceipt(
            action="skipped",
            reason_codes=("privacy_or_unconfigured",),
        )
        _LAST_STATIC_LINT.value = receipt.to_dict()
        return receipt
    from ipfs_accelerate_py.typesafe_inference import system_one

    try:
        result = system_one(
            {
                "obligation": {"id": str(obligation_id or "")[:128]},
                "span": {
                    "path": str(path or "")[:128],
                    "summary": str(summary or "")[:400],
                },
            },
            {
                "relevant_to_obligation": Noul(
                    instructions={
                        "question": "Is `span` relevant to `obligation.id`?",
                        "inspect": "`span.summary`",
                    },
                ),
                "review_needed": Score(
                    instructions={
                        "question": "How strongly should a human review this span?",
                    },
                    criteria=["none", "optional", "required"],
                ),
            },
            timeout=timeout,
        )
    except Exception:
        receipt = AdvisoryReceipt(action="abstain", reason_codes=("typesafe_error",))
        _LAST_STATIC_LINT.value = receipt.to_dict()
        return receipt
    nouls = getattr(result, "nouls", None) or {}
    scores = getattr(result, "scores", None) or {}
    relevant = float(getattr(nouls.get("relevant_to_obligation"), "noul", 0.0) or 0.0)
    review = float(getattr(scores.get("review_needed"), "score", 0.0) or 0.0)
    reasons = ["composed_in_code", "static_analyzer_still_authoritative"]
    if relevant < 0.4:
        reasons.append("low_obligation_relevance")
    if review >= 1.5:
        reasons.append("human_review_suggested")
    receipt = AdvisoryReceipt(
        action="linted",
        noul=relevant,
        score=review,
        reason_codes=tuple(reasons),
    )
    payload = receipt.to_dict()
    payload["accepted_as_authority"] = False
    payload["replaces_static_analysis"] = False
    _LAST_STATIC_LINT.value = payload
    return receipt


_LAST_REFACTOR_SCOPE = threading.local()


def last_refactor_scope() -> dict[str, Any]:
    value = getattr(_LAST_REFACTOR_SCOPE, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def observe_refactor_scope(
    *,
    declared_paths: Sequence[str] = (),
    changed_paths: Sequence[str] = (),
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Advisory view of declared vs changed paths. Does not replace undeclared-refactor."""

    declared = tuple(str(item).strip() for item in declared_paths if str(item).strip())[:8]
    changed = tuple(str(item).strip() for item in changed_paths if str(item).strip())[:8]
    payload = {
        "accepted_as_authority": False,
        "replaces_undeclared_refactor_check": False,
        "declared_paths": list(declared),
        "changed_paths": list(changed),
    }
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        _LAST_REFACTOR_SCOPE.value = dict(payload)
        return payload
    from ipfs_accelerate_py.typesafe_inference import system_one

    try:
        result = system_one(
            {"declared_paths": list(declared), "changed_paths": list(changed)},
            {
                "in_declared_scope": Noul(
                    instructions={
                        "question": "Are `changed_paths` inside `declared_paths`?",
                        "compare": ["`changed_paths`", "`declared_paths`"],
                    },
                    criteria=noul_yes_no(
                        true_what="Every changed path is under a declared prefix",
                        true_examples=["src/a.py under src/"],
                        false_what="A changed path sits outside declared prefixes",
                        false_examples=["docs/secret.md when declared is src/"],
                    ),
                ),
            },
            timeout=timeout,
        )
    except Exception:
        _LAST_REFACTOR_SCOPE.value = dict(payload)
        return payload
    noul = getattr(
        (getattr(result, "nouls", None) or {}).get("in_declared_scope"),
        "noul",
        0.0,
    )
    payload["in_declared_scope"] = round(float(noul or 0.0), 4)
    _LAST_REFACTOR_SCOPE.value = dict(payload)
    return payload


_LAST_MERGE_CONFLICT = threading.local()


def last_merge_conflict() -> dict[str, Any]:
    value = getattr(_LAST_MERGE_CONFLICT, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def observe_merge_conflict_paths(
    *,
    declared_paths: Sequence[str] = (),
    conflict_paths: Sequence[str] = (),
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Advisory noul: are conflict paths inside declared scope?

    Never fences merge-queue consumer_id. Never writes the merge.
    """

    view = observe_refactor_scope(
        declared_paths=declared_paths,
        changed_paths=conflict_paths,
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
        timeout=timeout,
    )
    payload = dict(view)
    payload["accepted_as_authority"] = False
    payload["replaces_consumer_fence"] = False
    payload["writes_merge"] = False
    payload["conflict_paths"] = list(payload.get("changed_paths") or [])
    pick = observe_extracted_span(
        " ".join(payload["conflict_paths"]),
        payload["conflict_paths"],
    )
    payload["typesafe_pick"] = {
        "pick": pick.get("pick") or "",
        "invents_span": False,
        "accepted_as_authority": False,
    }
    if payload["typesafe_pick"]["pick"] not in set(payload["conflict_paths"]) | {
        "",
        PICK_NONE,
    }:
        payload["typesafe_pick"]["pick"] = PICK_NONE
    _LAST_MERGE_CONFLICT.value = dict(payload)
    return payload


_LAST_PRODUCER_CONSUMER = threading.local()


def last_producer_consumer() -> dict[str, Any]:
    value = getattr(_LAST_PRODUCER_CONSUMER, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def producer_consumer_questions() -> dict[str, Any]:
    return {
        "producer_matches": Noul(
            instructions={
                "field": {
                    "name": "producer",
                    "type": "string",
                    "description": "Admitted AST extractor identity",
                },
                "question": "Does `artifact.producer` match `admitted.producer`?",
                "compare": ["`artifact.producer`", "`admitted.producer`"],
            },
            criteria=noul_yes_no(
                true_what="The claimed producer identity equals the admitted producer",
                true_examples=["typescript-compiler-api vs typescript-compiler-api"],
                false_what="The claimed producer is a different extractor or empty",
                false_examples=["regex-heuristic vs typescript-compiler-api"],
            ),
        ),
        "producer_version_matches": Noul(
            instructions={
                "field": {
                    "name": "producer_version",
                    "type": "string",
                    "description": "Admitted AST extractor version string",
                },
                "question": (
                    "Does `artifact.producer_version` match `admitted.producer_version`?"
                ),
                "compare": [
                    "`artifact.producer_version`",
                    "`admitted.producer_version`",
                ],
            },
            criteria=noul_yes_no(
                true_what="Extractor versions are the same admitted string",
                true_examples=["typescript-ast-extractor@2 vs typescript-ast-extractor@2"],
                false_what="Versions differ or one side is missing",
                false_examples=["@1 vs @2"],
            ),
        ),
    }


def lint_producer_consumer(
    *,
    artifact_id: str = "",
    claimed_producer: str = "",
    admitted_producer: str = "",
    claimed_producer_version: str = "",
    admitted_producer_version: str = "",
    claimed_consumer_id: str = "",
    admitted_consumer_id: str = "",
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Advisory noul: does this artifact's producer match the admitted producer?

    Never rewrites ``producer_id``. Never replaces polyglot ``PROTOCOL_ERROR``.
    Never fences merge-queue ``consumer_id``.
    """

    payload = {
        "accepted_as_authority": False,
        "rewrites_producer_id": False,
        "replaces_protocol_error": False,
        "replaces_consumer_fence": False,
        "artifact_id": str(artifact_id or "")[:128],
        "claimed_producer": str(claimed_producer or "")[:128],
        "admitted_producer": str(admitted_producer or "")[:128],
        "claimed_producer_version": str(claimed_producer_version or "")[:128],
        "admitted_producer_version": str(admitted_producer_version or "")[:128],
        "claimed_consumer_id": str(claimed_consumer_id or "")[:128],
        "admitted_consumer_id": str(admitted_consumer_id or "")[:128],
    }
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        _LAST_PRODUCER_CONSUMER.value = dict(payload)
        return payload
    from ipfs_accelerate_py.typesafe_inference import system_one

    try:
        result = system_one(
            {
                "artifact": {
                    "id": payload["artifact_id"],
                    "producer": payload["claimed_producer"],
                    "producer_version": payload["claimed_producer_version"],
                    "consumer_id": payload["claimed_consumer_id"],
                },
                "admitted": {
                    "producer": payload["admitted_producer"],
                    "producer_version": payload["admitted_producer_version"],
                    "consumer_id": payload["admitted_consumer_id"],
                },
            },
            producer_consumer_questions(),
            timeout=timeout,
        )
    except Exception:
        _LAST_PRODUCER_CONSUMER.value = dict(payload)
        return payload
    nouls = getattr(result, "nouls", None) or {}
    producer_noul = float(
        getattr(nouls.get("producer_matches"), "noul", 0.0) or 0.0
    )
    version_noul = float(
        getattr(nouls.get("producer_version_matches"), "noul", 0.0) or 0.0
    )
    payload["producer_matches"] = round(producer_noul, 4)
    payload["producer_version_matches"] = round(version_noul, 4)
    reasons = ["composed_in_code", "advisory_lint_only"]
    if producer_noul < 0.4:
        reasons.append("producer_mismatch")
    if version_noul < 0.4 and (
        payload["claimed_producer_version"] or payload["admitted_producer_version"]
    ):
        reasons.append("producer_version_mismatch")
    payload["reason_codes"] = reasons
    _LAST_PRODUCER_CONSUMER.value = dict(payload)
    return payload


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
            criteria=noul_yes_no(
                true_what="A listed receipt id backs the KERNEL_VERIFIED or test claim",
                true_examples=["KERNEL_VERIFIED with receipt-1 in receipt_ids"],
                false_what="The claim asserts KERNEL_VERIFIED with no matching receipt",
                false_examples=["KERNEL_VERIFIED everything with empty receipt_ids"],
            ),
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

    rows = selected[:8]
    state = {
        "receipt_ids": list(receipts[:16]),
        "claims": {
            str(row.get("id") or "").strip(): {
                "id": str(row.get("id") or "").strip(),
                "text": str(row.get("text") or "")[:240],
            }
            for row in rows
            if str(row.get("id") or "").strip()
        },
    }
    questions: dict[str, Any] = {}
    for ident in state["claims"]:
        questions[f"supported_{ident}"] = Noul(
            instructions={
                "question": (
                    f"Is `claims.{ident}.text` supported by one of `receipt_ids`?"
                ),
                "compare": [f"`claims.{ident}.text`", "`receipt_ids`"],
                "focus": "KERNEL_VERIFIED or test claims need a matching receipt id.",
            },
            criteria=citation_questions()["supported"].criteria,
        )
    try:
        result = system_one(state, questions, timeout=timeout)
    except Exception:
        return ()
    nouls = getattr(result, "nouls", None) or {}
    unsupported: list[str] = []
    for ident in state["claims"]:
        noul = getattr(nouls.get(f"supported_{ident}"), "noul", 1.0)
        try:
            supported = float(noul or 0.0)
        except (TypeError, ValueError):
            supported = 1.0
        if supported < 0.4:
            unsupported.append(ident)
    return tuple(unsupported)


def last_claim_citation() -> dict[str, Any]:
    value = getattr(_LAST_CITATION, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_citation_text(text: str) -> str:
    """Collapse whitespace and fold curly quotes so a quote matches across wraps."""

    return re.sub(r"\s+", " ", str(text or "").translate(_CURLY_QUOTES)).strip()


def _mirror_claim_citation(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
            mirror_work_record,
        )

        record_ref = str(
            payload.get("verdict")
            or payload.get("status")
            or (payload.get("reason_codes") or ["claim-citation"])[0]
            or "claim-citation"
        )
        mirror_work_record(
            catalog_kind="metadata",
            record_kind="claim_citation",
            record_ref=record_ref,
            subject_kind="record_cid",
            subject_ref=record_ref,
        )
    except Exception:
        pass
    return payload


def check_claim_citation(
    source: str,
    claim: str,
    *,
    quote: str = "",
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
    auto_accept: float = CITATION_AUTO_ACCEPT,
) -> dict[str, Any]:
    """String-match the quote first, then Choice how the source relates to the claim.

    A missing quote is ``fabricated`` with no HTTP. Cookbook ``verified`` is stored
    as ``supports`` — never KERNEL_VERIFIED.
    """

    payload: dict[str, Any] = {
        "accepted_as_authority": False,
        "rewrites_ir": False,
        "kernel_verified": False,
        "verdict": "",
        "confidence": 0.0,
        "auto": False,
        "status": "privacy_or_unconfigured",
        "reason_codes": ["privacy_or_unconfigured"],
    }
    needle = _normalize_citation_text(quote)
    haystack = _normalize_citation_text(source)
    if needle:
        if not haystack or needle not in haystack:
            payload["status"] = "missing"
            payload["verdict"] = "fabricated"
            payload["auto"] = True
            payload["confidence"] = None
            payload["reason_codes"] = ["string_match", "fabricated_no_http"]
            _LAST_CITATION.value = dict(payload)
            return _mirror_claim_citation(payload)
        payload["status"] = "found"
    else:
        payload["status"] = "section-only"
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        _LAST_CITATION.value = dict(payload)
        return _mirror_claim_citation(payload)
    from ipfs_accelerate_py.typesafe_inference import Choice, system_one

    try:
        result = system_one(
            {
                "claim": str(claim or "")[:240],
                "section": str(source or "")[:800],
            },
            {
                "relation": Choice(
                    instructions={
                        "question": "How does the section relate to the claim?",
                        "compare": ["`section`", "`claim`"],
                    },
                    criteria={
                        "supports": {
                            "what": (
                                "The section states the claim or directly "
                                "implies that it is true"
                            )
                        },
                        "contradicts": {
                            "what": (
                                "The section states the opposite of the claim "
                                "or implies it is false"
                            )
                        },
                        "says_nothing": {
                            "what": (
                                "The section does not address what the claim "
                                "asserts, either way"
                            )
                        },
                    },
                )
            },
            timeout=timeout,
        )
    except Exception:
        payload["reason_codes"] = ["typesafe_error_fail_open"]
        _LAST_CITATION.value = dict(payload)
        return _mirror_claim_citation(payload)
    answer = (getattr(result, "choices", None) or {}).get("relation")
    picked = str(getattr(answer, "choice", "") or "").strip()
    if picked not in CITATION_CHOICES:
        picked = "says_nothing"
    conf = float(getattr(answer, "confidence", 0.0) or 0.0)
    payload["verdict"] = picked
    payload["confidence"] = round(conf, 4)
    payload["auto"] = conf >= float(auto_accept)
    payload["reason_codes"] = ["composed_in_code", "citation_check", "advisory_only"]
    _LAST_CITATION.value = dict(payload)
    return _mirror_claim_citation(payload)


def observe_claim_citation(
    source: str,
    claim: str,
    *,
    quote: str = "",
) -> dict[str, Any]:
    """Never-raises wrapper. Does not admit KERNEL_VERIFIED."""

    try:
        return check_claim_citation(source, claim, quote=quote)
    except Exception:
        payload = {
            "accepted_as_authority": False,
            "kernel_verified": False,
            "verdict": "",
            "auto": False,
        }
        _LAST_CITATION.value = dict(payload)
        return payload


def last_extracted_span() -> dict[str, Any]:
    value = getattr(_LAST_PICK, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def pick_extracted_span(
    clause: str,
    candidates: Sequence[str],
    *,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Choice among regex-extracted spans plus ``none``.

    Copies the pick verbatim. Never invents a name. Callers keep the full
    extracted set; this is sidecar metadata only.
    """

    ordered = tuple(
        dict.fromkeys(
            str(item).strip()
            for item in candidates
            if str(item).strip() and str(item).strip().casefold() != PICK_NONE
        )
    )[:PICK_MAX]
    payload: dict[str, Any] = {
        "accepted_as_authority": False,
        "rewrites_ir": False,
        "invents_span": False,
        "pick": "",
        "candidates": list(ordered),
        "confidence": 0.0,
        "reason_codes": ["privacy_or_unconfigured"],
    }
    if not ordered:
        payload["reason_codes"] = ["no_candidates"]
        _LAST_PICK.value = dict(payload)
        return payload
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        _LAST_PICK.value = dict(payload)
        return payload
    from ipfs_accelerate_py.typesafe_inference import Choice, system_one

    criteria = {
        ident: {"what": ident, "not_for": "any other listed span"}
        for ident in ordered
    }
    criteria[PICK_NONE] = {"what": "None of these is the requested value."}
    try:
        result = system_one(
            {
                "clause": str(clause or "")[:240],
                "candidates": list(ordered),
            },
            {
                "pick": Choice(
                    instructions={
                        "question": (
                            "Which extracted span is the primary path or "
                            "claim named by `clause`?"
                        ),
                        "inspect": "`clause`",
                    },
                    criteria=criteria,
                )
            },
            timeout=timeout,
        )
    except Exception:
        payload["reason_codes"] = ["typesafe_error_fail_open"]
        _LAST_PICK.value = dict(payload)
        return payload
    answer = (getattr(result, "choices", None) or {}).get("pick")
    picked = str(getattr(answer, "choice", "") or "").strip()
    conf = float(getattr(answer, "confidence", 0.0) or 0.0)
    payload["confidence"] = round(conf, 4)
    allowed = set(ordered) | {PICK_NONE}
    if picked not in allowed:
        payload["pick"] = PICK_NONE
        payload["reason_codes"] = ["composed_in_code", "unknown_choice"]
    else:
        payload["pick"] = picked
        payload["reason_codes"] = ["composed_in_code", "pre_parsed_pick"]
    _LAST_PICK.value = dict(payload)
    return payload


def observe_extracted_span(
    clause: str,
    candidates: Sequence[str],
) -> dict[str, Any]:
    """Never-raises wrapper. Does not rewrite the extracted set."""

    try:
        return pick_extracted_span(clause, candidates)
    except Exception:
        payload = {
            "accepted_as_authority": False,
            "invents_span": False,
            "pick": "",
        }
        _LAST_PICK.value = dict(payload)
        return payload


def last_clause_date() -> dict[str, Any]:
    value = getattr(_LAST_DATE, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _years_in_clause(clause: str) -> tuple[str, ...]:
    found: list[str] = []
    for match in _YEAR_IN_TEXT.findall(str(clause or "")):
        year = int(match)
        if DATE_YEAR_MIN <= year <= DATE_YEAR_MAX and match not in found:
            found.append(match)
        if len(found) >= PICK_MAX:
            break
    return tuple(found)


def _date_part(parts: Mapping[str, Any], key: str) -> tuple[str, float]:
    raw = parts.get(key) if isinstance(parts.get(key), Mapping) else {}
    choice = str((raw or {}).get("choice") or "none").strip() or "none"
    conf = float((raw or {}).get("confidence") or 0.0)
    return choice, conf


def assemble_date_parts(
    parts: Mapping[str, Any],
    *,
    today: date,
) -> dict[str, Any]:
    """Turn TypeSafe date *parts* into a calendar date in Python.

    The model never adds numbers or does weekday arithmetic.
    """

    mode, mode_conf = _date_part(parts, "mode")
    confs = [mode_conf]

    def _result(resolved: date | None, note: str) -> dict[str, Any]:
        usable = [float(item) for item in confs]
        confidence = min(usable) if usable else 0.0
        incomplete = resolved is None
        needs_review = incomplete or confidence < DATE_REVIEW_BELOW
        return {
            "date": resolved.isoformat() if resolved is not None else "",
            "mode": mode,
            "confidence": round(confidence, 4),
            "needs_review": needs_review,
            "incomplete": incomplete,
            "note": note,
        }

    if mode == "none":
        return _result(None, "no such date stated")
    if mode == "absolute":
        month, month_conf = _date_part(parts, "month")
        day, day_conf = _date_part(parts, "day")
        year, year_conf = _date_part(parts, "year")
        confs += [month_conf, day_conf, year_conf]
        if month not in DATE_MONTHS or day in {"none", ""} or not day.isdigit():
            return _result(None, "absolute date incomplete")
        if year == "out_of_range":
            return _result(None, f"year outside {DATE_YEAR_MIN}-{DATE_YEAR_MAX}")
        month_num = DATE_MONTHS[month]
        day_num = int(day)
        if year == "none":
            try:
                resolved = date(today.year, month_num, day_num)
            except ValueError:
                return _result(None, f"impossible date: {month} {day}")
            if resolved < today - timedelta(days=31):
                try:
                    resolved = date(today.year + 1, month_num, day_num)
                except ValueError:
                    return _result(None, f"impossible date: {month} {day}")
            return _result(resolved, "")
        if not year.isdigit():
            return _result(None, "absolute date incomplete")
        try:
            return _result(date(int(year), month_num, day_num), "")
        except ValueError:
            return _result(None, f"impossible date: {year}-{month}-{day}")
    if mode == "relative":
        anchor, anchor_conf = _date_part(parts, "day_anchor")
        confs.append(anchor_conf)
        if anchor == "today":
            return _result(today, "")
        if anchor == "tomorrow":
            return _result(today + timedelta(days=1), "")
        if anchor == "day_after":
            return _result(today + timedelta(days=2), "")
        if anchor == "weekday":
            weekday, weekday_conf = _date_part(parts, "weekday")
            offset, offset_conf = _date_part(parts, "week_offset")
            confs += [weekday_conf, offset_conf]
            if weekday not in DATE_WEEKDAYS:
                return _result(None, "relative weekday not read")
            weekday_index = DATE_WEEKDAYS.index(weekday)
            this_monday = today - timedelta(days=today.weekday())
            if offset == "next":
                resolved = this_monday + timedelta(days=7 + weekday_index)
            elif offset == "current":
                resolved = this_monday + timedelta(days=weekday_index)
            else:
                resolved = today + timedelta(
                    days=(weekday_index - today.weekday()) % 7
                )
            return _result(resolved, "")
        return _result(None, "relative day not read")
    return _result(None, f"unrecognized mode: {mode}")


def extract_clause_date(
    clause: str,
    *,
    role: str = "the primary date stated in the clause",
    today: date | None = None,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """TypeSafe reads date parts; code assembles the calendar date."""

    today = today or date.today()
    payload: dict[str, Any] = {
        "accepted_as_authority": False,
        "kernel_verified": False,
        "date": "",
        "mode": "",
        "confidence": 0.0,
        "needs_review": False,
        "incomplete": False,
        "note": "",
        "parts": {},
        "reason_codes": ["privacy_or_unconfigured"],
    }
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        _LAST_DATE.value = dict(payload)
        return payload
    from ipfs_accelerate_py.typesafe_inference import Choice, system_one

    absent = "The document does not state this, or it is not this kind of date."
    years = _years_in_clause(clause)
    year_criteria = {year: {"what": year} for year in years}
    year_criteria["out_of_range"] = {
        "what": f"A year is stated but is outside {DATE_YEAR_MIN}-{DATE_YEAR_MAX}"
    }
    year_criteria["none"] = {"what": "No year is stated for this date."}
    labeled = str(role or "the primary date stated in the clause").strip()[:80]
    questions = {
        "mode": Choice(
            instructions={
                "question": (
                    f"How is {labeled} written? absolute names a month; "
                    "relative is today/tomorrow/weekday; none if unstated."
                ),
                "inspect": "`clause`",
            },
            criteria={
                "absolute": {"what": "A calendar date naming a month"},
                "relative": {"what": "A date relative to today"},
                "none": {"what": "The document does not state this date"},
            },
        ),
        "month": Choice(
            instructions={
                "question": f"If {labeled} is absolute, which month?",
                "inspect": "`clause`",
            },
            criteria={
                **{name: {"what": name} for name in DATE_MONTHS},
                "none": {"what": absent},
            },
        ),
        "day": Choice(
            instructions={
                "question": (
                    f"If {labeled} is absolute, which day of the month (1-31)?"
                ),
                "inspect": "`clause`",
            },
            criteria={
                **{str(day): {"what": str(day)} for day in range(1, 32)},
                "none": {"what": absent},
            },
        ),
        "year": Choice(
            instructions={
                "question": f"If {labeled} is absolute, which year?",
                "inspect": "`clause`",
            },
            criteria=year_criteria,
        ),
        "day_anchor": Choice(
            instructions={
                "question": (
                    f"If {labeled} is relative, which day is it relative to today?"
                ),
                "inspect": "`clause`",
            },
            criteria={
                "today": {"what": "today"},
                "tomorrow": {"what": "tomorrow"},
                "day_after": {"what": "the day after tomorrow"},
                "weekday": {"what": "a named weekday"},
                "none": {"what": absent},
            },
        ),
        "weekday": Choice(
            instructions={
                "question": f"If {labeled} names a weekday, which one?",
                "inspect": "`clause`",
            },
            criteria={
                **{name: {"what": name} for name in DATE_WEEKDAYS},
                "none": {"what": absent},
            },
        ),
        "week_offset": Choice(
            instructions={
                "question": (
                    f"If {labeled} names a weekday, which week: current, next, or none?"
                ),
                "inspect": "`clause`",
            },
            criteria={
                "current": {"what": "this week"},
                "next": {"what": "next week"},
                "none": {"what": absent},
            },
        ),
    }
    try:
        result = system_one(
            {"clause": str(clause or "")[:240], "role": labeled},
            questions,
            timeout=timeout,
        )
    except Exception:
        payload["reason_codes"] = ["typesafe_error_fail_open"]
        _LAST_DATE.value = dict(payload)
        return payload
    parts: dict[str, dict[str, Any]] = {}
    choices = getattr(result, "choices", None) or {}
    for key in questions:
        answer = choices.get(key)
        parts[key] = {
            "choice": str(getattr(answer, "choice", "") or "").strip() or "none",
            "confidence": round(
                float(getattr(answer, "confidence", 0.0) or 0.0), 4
            ),
        }
    assembled = assemble_date_parts(parts, today=today)
    payload["parts"] = parts
    payload["date"] = assembled["date"]
    payload["mode"] = assembled["mode"]
    payload["confidence"] = assembled["confidence"]
    payload["needs_review"] = assembled["needs_review"]
    payload["incomplete"] = assembled["incomplete"]
    payload["note"] = assembled["note"]
    payload["reason_codes"] = ["composed_in_code", "date_parts_only"]
    _LAST_DATE.value = dict(payload)
    return payload


def observe_clause_date(
    clause: str,
    *,
    role: str = "the primary date stated in the clause",
    today: date | None = None,
) -> dict[str, Any]:
    """Never-raises wrapper. Does not rewrite traces or admit proofs."""

    try:
        return extract_clause_date(clause, role=role, today=today)
    except Exception:
        payload = {
            "accepted_as_authority": False,
            "kernel_verified": False,
            "date": "",
            "incomplete": False,
        }
        _LAST_DATE.value = dict(payload)
        return payload


def last_supporting_line() -> dict[str, Any]:
    value = getattr(_LAST_FIND, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _tagged_source_lines(
    text: str, *, limit: int = FIND_MAX_LINES
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in str(text or "").splitlines():
        stripped = re.sub(r"[\t ]+", " ", raw).strip()
        if not stripped:
            continue
        rows.append((f"L{len(rows):03d}", stripped))
        if len(rows) >= limit:
            break
    return rows


def find_supporting_line(
    source: str,
    query: str,
    *,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Choice over existing line ids plus an exists noul.

    Ranking cannot invent a line. ``exists`` decides whether any line answers
    ``query``. Text is copied from the source only.
    """

    rows = _tagged_source_lines(source)
    payload: dict[str, Any] = {
        "accepted_as_authority": False,
        "kernel_verified": False,
        "invents_ids": False,
        "line_id": "",
        "line_text": "",
        "exists": 0.0,
        "verdict": "",
        "ranked": [],
        "reason_codes": ["privacy_or_unconfigured"],
    }
    if not rows:
        payload["reason_codes"] = ["no_lines"]
        _LAST_FIND.value = dict(payload)
        return payload
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        _LAST_FIND.value = dict(payload)
        return payload
    from ipfs_accelerate_py.typesafe_inference import Choice, Noul, system_one

    ids = [ident for ident, _text in rows]
    by_id = {ident: text for ident, text in rows}
    tagged = "\n".join(f"{ident}| {text}" for ident, text in rows)
    ask = str(query or "")[:240]
    try:
        result = system_one(
            {"lines": tagged, "query": ask},
            {
                "where": Choice(
                    instructions={
                        "question": (
                            "Which line of `lines` contains the answer to `query`?"
                        ),
                        "inspect": "`lines`",
                    },
                    criteria={ident: {"what": ident} for ident in ids},
                ),
                "exists": Noul(
                    instructions={
                        "question": (
                            "Does any line of `lines` address or answer `query`?"
                        ),
                        "inspect": "`lines`",
                    },
                ),
            },
            timeout=timeout,
        )
    except Exception:
        payload["reason_codes"] = ["typesafe_error_fail_open"]
        _LAST_FIND.value = dict(payload)
        return payload
    where = (getattr(result, "choices", None) or {}).get("where")
    picked = str(getattr(where, "choice", "") or "").strip()
    if picked not in by_id:
        picked = ""
    probs = dict(getattr(where, "probabilities", None) or {})
    ranked_ids = sorted(
        ids,
        key=lambda ident: (
            -float(probs.get(ident, 0.0) or 0.0),
            ids.index(ident),
        ),
    )
    if not probs and picked:
        ranked_ids = [picked] + [ident for ident in ids if ident != picked]
    ranked = [
        {
            "id": ident,
            "score": round(float(probs.get(ident, 0.0) or 0.0), 4),
            "text": by_id[ident],
        }
        for ident in ranked_ids[:4]
    ]
    exists = float(
        getattr((getattr(result, "nouls", None) or {}).get("exists"), "noul", 0.0)
        or 0.0
    )
    if exists >= FIND_EXISTS_HIGH:
        verdict = "answered"
    elif exists < FIND_EXISTS_LOW:
        verdict = "absent"
    else:
        verdict = "partial"
    payload["line_id"] = picked
    payload["line_text"] = by_id.get(picked, "")
    payload["exists"] = round(exists, 4)
    payload["verdict"] = verdict
    payload["ranked"] = ranked
    payload["reason_codes"] = ["composed_in_code", "line_ids_from_source"]
    _LAST_FIND.value = dict(payload)
    return payload


def observe_supporting_line(source: str, query: str) -> dict[str, Any]:
    """Never-raises wrapper. Does not rewrite traces or admit a source span."""

    try:
        return find_supporting_line(source, query)
    except Exception:
        payload = {
            "accepted_as_authority": False,
            "kernel_verified": False,
            "invents_ids": False,
            "line_id": "",
            "verdict": "",
        }
        _LAST_FIND.value = dict(payload)
        return payload


def last_line_stitch() -> dict[str, Any]:
    value = getattr(_LAST_STITCH, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def stitch_hard_wrapped_lines(
    text: str,
    *,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Merge hard-wrapped lines. Output characters come only from the input.

    Fail-open keeps the original text. Never generates formulas or markup.
    """

    original = str(text or "")
    payload: dict[str, Any] = {
        "accepted_as_authority": False,
        "kernel_verified": False,
        "generates_text": False,
        "generates_markup": False,
        "text": original,
        "original": original,
        "blocks": [],
        "reason_codes": ["privacy_or_unconfigured"],
    }
    raw_lines = original.split("\n")
    lines: list[dict[str, Any]] = []
    gap = False
    for raw in raw_lines:
        stripped = re.sub(r"[\t ]+", " ", raw).strip()
        if not stripped:
            gap = bool(lines)
            continue
        lines.append({"text": stripped, "gap": gap})
        gap = False
        if len(lines) >= 17:
            break
    if len(lines) < 2 or not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        _LAST_STITCH.value = dict(payload)
        return payload
    from ipfs_accelerate_py.typesafe_inference import Noul, system_one

    questions: dict[str, Any] = {}
    for index in range(1, len(lines)):
        if lines[index]["gap"]:
            continue
        ident = f"L{index:03d}"
        questions[ident] = Noul(
            instructions={
                "question": (
                    f"Does line L{index:03d} pick up mid-sentence, continuing "
                    f"a sentence left unfinished at the end of line L{index - 1:03d}?"
                ),
            },
        )
    if not questions:
        _LAST_STITCH.value = dict(payload)
        return payload
    tagged = "\n".join(
        f"{chr(10) if item['gap'] else ''}L{i:03d}| {item['text']}"
        for i, item in enumerate(lines)
    )
    try:
        result = system_one({"lines": tagged}, questions, timeout=timeout)
    except Exception:
        payload["reason_codes"] = ["typesafe_error_fail_open"]
        _LAST_STITCH.value = dict(payload)
        return payload
    nouls = getattr(result, "nouls", None) or {}
    joins = [0.0] * len(lines)
    for index in range(1, len(lines)):
        ident = f"L{index:03d}"
        joins[index] = float(getattr(nouls.get(ident), "noul", 0.0) or 0.0)
    block_rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        bar = (
            JOIN_AFTER_TERMINAL
            if index and _TERMINAL_END.search(lines[index - 1]["text"])
            else JOIN_AFTER_DANGLING
        )
        if block_rows and not line["gap"] and joins[index] >= bar:
            block_rows[-1]["text"] += " " + line["text"]
        else:
            block_rows.append({"text": line["text"], "gap": bool(line["gap"])})
    merged = "\n".join(item["text"] for item in block_rows)
    payload["text"] = merged
    payload["reason_codes"] = ["composed_in_code", "characters_from_input"]
    payload["blocks"] = _classify_stitched_blocks(block_rows, timeout=timeout)
    _LAST_STITCH.value = dict(payload)
    return payload


def _classify_stitched_blocks(
    block_rows: Sequence[Mapping[str, Any]],
    *,
    timeout: float = 15.0,
) -> list[dict[str, Any]]:
    """Pass-2 Choice over stitched blocks. Labels only; characters stay from input."""

    rows = list(block_rows)[:CLASSIFY_MAX_BLOCKS]
    labeled = [
        {
            "id": f"B{index:03d}",
            "text": str(item.get("text") or ""),
            "type": "",
            "confidence": 0.0,
            "hlevel": "",
            "step": 0.0,
            "ordered": False,
            "callout": "",
        }
        for index, item in enumerate(rows)
        if str(item.get("text") or "").strip()
    ]
    if not labeled:
        return []
    try:
        from ipfs_accelerate_py.typesafe_inference import Choice, Noul, system_one
    except Exception:
        return labeled
    questions: dict[str, Any] = {}
    for item in labeled:
        bid = item["id"]
        questions[f"type_{bid}"] = Choice(
            instructions={
                "question": f"What kind of content is block {bid}?",
                "inspect": "`blocks`",
            },
            criteria={
                "heading": {
                    "what": "A short label or title, not a full sentence of content"
                },
                "paragraph": {
                    "what": "Running prose of one or more complete sentences"
                },
                "list_item": {"what": "One entry in a list of parallel items"},
                "quote": {"what": "Words attributed to a person or source"},
                "code": {"what": "Code, a shell command, or a config snippet"},
                "callout": {
                    "what": "A warning, tip, or note set apart from the main text"
                },
            },
        )
        if len(item["text"]) <= HEADING_MAX_CHARS:
            questions[f"hlevel_{bid}"] = Choice(
                instructions={
                    "question": (
                        f"As a heading, what level would block {bid} occupy?"
                    )
                },
                criteria={
                    "title": {"what": "The title of the whole document"},
                    "section": {"what": "A major section heading"},
                    "subsection": {"what": "A minor heading under a section"},
                },
            )
        questions[f"step_{bid}"] = Noul(
            instructions={
                "question": (
                    f"Is block {bid} a step in a sequence where order matters?"
                )
            },
        )
        questions[f"callout_{bid}"] = Choice(
            instructions={"question": f"What kind of aside is block {bid}?"},
            criteria={
                "note": {"what": "Neutral extra information"},
                "tip": {"what": "A helpful suggestion"},
                "warning": {"what": "A caution about harm or failure"},
            },
        )
    tagged = "\n".join(f"{item['id']}| {item['text']}" for item in labeled)
    try:
        result = system_one({"blocks": tagged}, questions, timeout=timeout)
    except Exception:
        return labeled
    choices = getattr(result, "choices", None) or {}
    nouls = getattr(result, "nouls", None) or {}
    for item in labeled:
        bid = item["id"]
        type_answer = choices.get(f"type_{bid}")
        picked = str(getattr(type_answer, "choice", "") or "").strip()
        item["type"] = picked if picked in BLOCK_TYPES else "paragraph"
        item["confidence"] = round(
            float(getattr(type_answer, "confidence", 0.0) or 0.0), 4
        )
        if item["type"] == "heading":
            level = str(
                getattr(choices.get(f"hlevel_{bid}"), "choice", "") or ""
            ).strip()
            item["hlevel"] = level if level in HEADING_LEVELS else "section"
        if item["type"] == "list_item":
            item["step"] = round(
                float(getattr(nouls.get(f"step_{bid}"), "noul", 0.0) or 0.0), 4
            )
            item["ordered"] = item["step"] >= STEP_THRESHOLD
        if item["type"] == "callout":
            kind = str(
                getattr(choices.get(f"callout_{bid}"), "choice", "") or ""
            ).strip()
            item["callout"] = kind if kind in CALLOUT_KINDS else "note"
    return labeled


def observe_line_stitch(text: str) -> dict[str, Any]:
    """Never-raises wrapper. Does not rewrite traces or generate markup."""

    try:
        return stitch_hard_wrapped_lines(text)
    except Exception:
        payload = {
            "accepted_as_authority": False,
            "kernel_verified": False,
            "generates_text": False,
            "generates_markup": False,
            "text": str(text or ""),
            "original": str(text or ""),
            "blocks": [],
        }
        _LAST_STITCH.value = dict(payload)
        return payload


__all__ = [
    "assemble_date_parts",
    "check_claim_citation",
    "cite_claim_spans",
    "citation_questions",
    "extract_clause_date",
    "find_supporting_line",
    "last_claim_citation",
    "last_clause_date",
    "last_extracted_span",
    "last_line_stitch",
    "last_supporting_line",
    "observe_claim_citation",
    "observe_clause_date",
    "observe_extracted_span",
    "observe_line_stitch",
    "observe_supporting_line",
    "pick_extracted_span",
    "stitch_hard_wrapped_lines",
    "compose_pairwise_order",
    "compose_snippet_score",
    "extract_claim_spans",
    "inspect_allowlisted_artifacts",
    "last_artifact_rank",
    "last_artifact_view",
    "last_merge_conflict",
    "last_parser_triage",
    "last_producer_consumer",
    "last_source_edit_lint",
    "last_refactor_scope",
    "last_static_lint",
    "lint_admissibility",
    "lint_producer_consumer",
    "observe_refactor_scope",
    "lint_static_span",
    "lint_questions",
    "observe_merge_conflict_paths",
    "observe_parser_failure_clusters",
    "observe_source_edit_lint",
    "pairwise_questions",
    "prepare_evidence_for_compile",
    "producer_consumer_questions",
    "rank_allowlisted_artifacts",
    "rerank_allowlisted_snippets",
    "rerank_questions",
]


def last_artifact_view() -> dict[str, Any]:
    value = getattr(_LAST_ARTIFACT_VIEW, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def inspect_allowlisted_artifacts(
    *,
    obligation_id: str = "",
    symbol_ids: Sequence[str] = (),
    clause_ids: Sequence[str] = (),
    summaries: Mapping[str, str] | None = None,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Read-only noul view of allowlisted AST/contract ids. Never writes."""

    symbols = tuple(str(item).strip() for item in symbol_ids if str(item).strip())[:8]
    clauses = tuple(str(item).strip() for item in clause_ids if str(item).strip())[:8]
    texts = {
        str(key).strip(): str(value)[:240]
        for key, value in dict(summaries or {}).items()
        if str(key).strip() in set(symbols) | set(clauses)
    }
    payload = {
        "accepted_as_authority": False,
        "writes_ast": False,
        "writes_contracts": False,
        "obligation_id": str(obligation_id or "")[:128],
        "symbol_ids": list(symbols),
        "clause_ids": list(clauses),
        "matches": {},
    }
    if not symbols and not clauses:
        _LAST_ARTIFACT_VIEW.value = dict(payload)
        return payload
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        _LAST_ARTIFACT_VIEW.value = dict(payload)
        return payload
    from ipfs_accelerate_py.typesafe_inference import system_one

    idents = tuple(dict.fromkeys((*symbols, *clauses)))
    state = {
        "obligation": {"id": str(obligation_id or "")[:128]},
        "artifacts": {
            ident: {
                "id": ident,
                "kind": "symbol" if ident in symbols else "clause",
                "summary": texts.get(ident, ""),
            }
            for ident in idents
        },
    }
    questions: dict[str, Any] = {
        f"matches_{ident}": Noul(
            instructions={
                "field": {
                    "name": "artifact_id",
                    "type": "string",
                    "description": "Allowlisted AST or ArchitectureIR node id",
                },
                "extracted_value": ident,
                "question": (
                    f"Does `artifacts.{ident}.summary` match `obligation.id`?"
                ),
                "inspect": f"`artifacts.{ident}.summary`",
            },
        )
        for ident in idents
    }
    questions.update(
        pairwise_questions(
            idents,
            left_path="artifacts.{id}.summary",
            right_path="artifacts.{id}.summary",
        )
    )
    try:
        result = system_one(state, questions, timeout=timeout)
    except Exception:
        _LAST_ARTIFACT_VIEW.value = dict(payload)
        return payload
    nouls = getattr(result, "nouls", None) or {}
    matches = {
        ident: round(float(getattr(nouls.get(f"matches_{ident}"), "noul", 0.0) or 0.0), 4)
        for ident in idents
    }
    payload["matches"] = matches
    payload["pair_nouls"] = {
        key: round(float(getattr(nouls.get(key), "noul", 0.5) or 0.5), 4)
        for key, _left, _right in _pairwise_keys(idents)
        if key in nouls
    }
    _LAST_ARTIFACT_VIEW.value = dict(payload)
    return payload


_LAST_ARTIFACT_RANK = threading.local()


def last_artifact_rank() -> dict[str, Any]:
    value = getattr(_LAST_ARTIFACT_RANK, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def rank_allowlisted_artifacts(
    artifact_ids: Sequence[str],
    *,
    obligation_id: str = "",
    summaries: Mapping[str, str] | None = None,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> tuple[str, ...]:
    """Order existing AST/IR ids. Fail-open to input order. Never invents ids.

    Ranking is non-probative. It does not replace undeclared-refactor checks
    or ArchitectureIR boundary cost ranking.
    """

    ordered: list[str] = []
    seen: set[str] = set()
    for item in artifact_ids:
        ident = str(item).strip()
        if not ident or ident in seen:
            continue
        seen.add(ident)
        ordered.append(ident)
        if len(ordered) >= MAX_SNIPPETS:
            break
    payload = {
        "accepted_as_authority": False,
        "invents_ids": False,
        "replaces_undeclared_refactor_check": False,
        "replaces_boundary_cost_ranking": False,
        "obligation_id": str(obligation_id or "")[:128],
        "ranked_ids": list(ordered),
        "matches": {},
        "shortlist": [],
        "fits": {},
        "suggested": "",
        "admits_candidate": False,
    }
    if not ordered:
        _LAST_ARTIFACT_RANK.value = dict(payload)
        return ()
    if not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        _LAST_ARTIFACT_RANK.value = dict(payload)
        return tuple(ordered)
    try:
        view = inspect_allowlisted_artifacts(
            obligation_id=obligation_id,
            symbol_ids=ordered,
            summaries=summaries,
            privacy_class=privacy_class,
            remote_disclosure_permitted=remote_disclosure_permitted,
            timeout=timeout,
        )
    except Exception:
        _LAST_ARTIFACT_RANK.value = dict(payload)
        return tuple(ordered)
    matches = {
        ident: float(score)
        for ident, score in dict(view.get("matches") or {}).items()
        if ident in seen
    }
    ranked = compose_pairwise_order(
        ordered,
        nouls=dict(view.get("pair_nouls") or {}),
        independent=matches,
    )
    payload["ranked_ids"] = list(ranked)
    payload["matches"] = matches
    payload["invents_ids"] = False
    texts = {
        ident: str((summaries or {}).get(ident) or ident)[:240] for ident in ranked
    }
    try:
        confirmed = _confirm_ranked_shortlist(
            obligation_id=str(obligation_id or "")[:128],
            shortlist=tuple(ranked[:RANK_SHORTLIST]),
            texts=texts,
            timeout=timeout,
        )
    except Exception:
        confirmed = {
            "shortlist": list(ranked[:RANK_SHORTLIST]),
            "fits": {},
            "suggested": "",
            "reason_codes": ["typesafe_error_fail_open"],
        }
    payload["shortlist"] = list(confirmed.get("shortlist") or [])
    payload["fits"] = dict(confirmed.get("fits") or {})
    payload["suggested"] = str(confirmed.get("suggested") or "")
    payload["admits_candidate"] = False
    payload["reason_codes"] = list(
        confirmed.get("reason_codes") or ["composed_in_code"]
    )
    _LAST_ARTIFACT_RANK.value = dict(payload)
    return tuple(ranked)


def _confirm_ranked_shortlist(
    *,
    obligation_id: str,
    shortlist: Sequence[str],
    texts: Mapping[str, str],
    timeout: float,
) -> dict[str, Any]:
    """Second pass: Choice over the shortlist plus ``fits::id`` nouls.

    If the best fits noul is under ``FITS_THRESHOLD``, suggest nothing.
    Never invents ids. Does not rewrite the first-pass ranking.
    """

    names = tuple(ident for ident in shortlist if str(ident).strip())[:RANK_SHORTLIST]
    payload: dict[str, Any] = {
        "shortlist": list(names),
        "fits": {},
        "suggested": "",
        "reason_codes": ["composed_in_code"],
    }
    if not names:
        payload["reason_codes"] = ["no_shortlist"]
        return payload
    try:
        from ipfs_accelerate_py.typesafe_inference import Choice, Noul, system_one
    except Exception:
        payload["reason_codes"] = ["typesafe_error_fail_open"]
        return payload
    questions: dict[str, Any] = {
        "which": Choice(
            instructions={
                "question": (
                    "Which shortlisted artifact is the right one for `obligation.id`?"
                ),
                "inspect": "`artifacts`",
            },
            criteria={
                ident: {"what": str(texts.get(ident) or ident)[:240]}
                for ident in names
            },
        )
    }
    for ident in names:
        questions[f"fits::{ident}"] = Noul(
            instructions={
                "question": (
                    f"Does artifact `{ident}` do the specific thing "
                    "`obligation.id` asks for?"
                ),
                "inspect": f"`artifacts.{ident}.summary`",
            },
        )
    try:
        result = system_one(
            {
                "obligation": {"id": obligation_id},
                "artifacts": {
                    ident: {
                        "id": ident,
                        "summary": str(texts.get(ident) or ident)[:240],
                    }
                    for ident in names
                },
            },
            questions,
            timeout=timeout,
        )
    except Exception:
        payload["reason_codes"] = ["typesafe_error_fail_open"]
        return payload
    nouls = getattr(result, "nouls", None) or {}
    fits = {
        ident: round(
            float(getattr(nouls.get(f"fits::{ident}"), "noul", 0.0) or 0.0), 4
        )
        for ident in names
    }
    payload["fits"] = fits
    best = max(fits.values()) if fits else 0.0
    if best < FITS_THRESHOLD:
        payload["reason_codes"] = ["composed_in_code", "nothing_fits"]
        return payload
    winner = str(
        getattr((getattr(result, "choices", None) or {}).get("which"), "choice", "")
        or ""
    ).strip()
    if winner not in names:
        payload["reason_codes"] = ["composed_in_code", "unknown_choice"]
        return payload
    payload["suggested"] = winner
    payload["reason_codes"] = ["composed_in_code", "shortlist_confirm"]
    return payload


_LAST_PARSER_TRIAGE = threading.local()


def last_parser_triage() -> dict[str, Any]:
    value = getattr(_LAST_PARSER_TRIAGE, "value", None)
    return dict(value) if isinstance(value, Mapping) else {}


def observe_parser_failure_clusters(
    clusters: Sequence[Mapping[str, Any]] = (),
    *,
    privacy_class: str = "repository_private",
    remote_disclosure_permitted: bool = True,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Advisory labels for parser-failure clusters. Never weakens thresholds.

    Protected MCP/runtime surfaces cannot be excluded by TypeSafe.
    """

    rows = tuple(dict(item) for item in clusters if isinstance(item, Mapping))[:8]
    protected = any(
        bool(item.get("protected") or item.get("protected_member_count"))
        for item in rows
    )
    payload = {
        "accepted_as_authority": False,
        "weakens_thresholds": False,
        "excludes_mcp_surface": False,
        "protected_contract_surface": protected,
        "cluster_ids": [str(item.get("cluster_id") or "")[:128] for item in rows],
        "fixture_like": {},
    }
    if protected:
        _LAST_PARSER_TRIAGE.value = dict(payload)
        return payload
    if not rows or not typesafe_permitted(
        privacy_class=privacy_class,
        remote_disclosure_permitted=remote_disclosure_permitted,
    ):
        _LAST_PARSER_TRIAGE.value = dict(payload)
        return payload
    from ipfs_accelerate_py.typesafe_inference import system_one

    clusters_state: dict[str, Any] = {}
    questions: dict[str, Any] = {}
    for item in rows:
        ident = str(item.get("cluster_id") or item.get("path_family") or "")[:128]
        if not ident:
            continue
        clusters_state[ident] = {
            "id": ident,
            "path_family": str(item.get("path_family") or "")[:128],
            "reason": str(item.get("reason_code") or "")[:64],
        }
        questions[f"fixture_or_generated_{ident}"] = Noul(
            instructions={
                "question": (
                    f"Is `clusters.{ident}.path_family` a generated or fixture "
                    "path rather than an MCP or runtime surface?"
                ),
                "inspect": f"`clusters.{ident}.path_family`",
            },
        )
    try:
        result = system_one({"clusters": clusters_state}, questions, timeout=timeout)
    except Exception:
        _LAST_PARSER_TRIAGE.value = dict(payload)
        return payload
    nouls = getattr(result, "nouls", None) or {}
    for ident in clusters_state:
        noul = getattr(nouls.get(f"fixture_or_generated_{ident}"), "noul", 0.0)
        payload["fixture_like"][ident] = round(float(noul or 0.0), 4)
    _LAST_PARSER_TRIAGE.value = dict(payload)
    return payload
