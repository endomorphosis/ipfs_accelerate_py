"""Advisory TypeSafe context rerank and admissibility lints.

Does not compile capsules, does not write patches, and does not drop
kernel-proved candidates. Unknown IDs are ignored. No key → original
snippet order and no lint flags.
"""

from __future__ import annotations

import re
import threading
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.typesafe_inference import Noul, Score, noul_yes_no
from ipfs_accelerate_py.agent_supervisor.integrations.typesafe_advisor import (
    AdvisoryReceipt,
    typesafe_permitted,
)

MAX_SNIPPETS = 8
MAX_SNIPPET_CHARS = 400
_LAST_SOURCE_EDIT_LINT = threading.local()
_LAST_ARTIFACT_VIEW = threading.local()


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


__all__ = [
    "cite_claim_spans",
    "citation_questions",
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
    _LAST_ARTIFACT_RANK.value = dict(payload)
    return tuple(ranked)


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
