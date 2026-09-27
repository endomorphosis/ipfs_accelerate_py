"""SPAR W4 CST-aware codemods on the existing board owner.

Deterministic CST transforms must preserve comments and source maps.
AST-only rewrites are not CST. ``libcst`` must not be claimed usable.
Undeclared CST transforms cannot complete a task. Missing payload is
fail-open. TypeSafe is never this owner.
"""

from __future__ import annotations

from typing import Any, Mapping

UNDECLARED_CST_TRANSFORM = "undeclared_cst_transform"
AST_ONLY_NOT_CST = "ast_only_not_cst"
LIBCST_NOT_USABLE = "libcst_not_usable"
COMMENTS_NOT_PRESERVED = "comments_not_preserved"
SOURCE_MAP_NOT_PRESERVED = "source_map_not_preserved"
CST_PRESERVED = "cst_preserved"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _record(state: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("cst_codemod", "codemod", "cst"):
        nested = state.get(key)
        if isinstance(nested, Mapping):
            return {**state, **nested}
    return state


def claims_cst_codemod(state: Mapping[str, Any] | None) -> bool:
    payload = _mapping(state)
    if isinstance(payload.get("cst_codemod"), Mapping) or isinstance(
        payload.get("codemod"), Mapping
    ):
        return True
    return bool(
        payload.get("cst_preserving") is True
        or payload.get("ast_only") is True
        or payload.get("libcst_usable") is True
        or payload.get("undeclared_cst") is True
        or payload.get("cst_profile")
    )


def cst_codemod_view(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Inspect a wake payload. Never completes a task."""

    payload = _mapping(state)
    claimed = claims_cst_codemod(payload)
    merged = _record(payload)
    ast_only = merged.get("ast_only") is True
    libcst = merged.get("libcst_usable") is True
    preserving = merged.get("cst_preserving") is True
    undeclared = merged.get("undeclared_cst") is True or (
        claimed
        and not preserving
        and not ast_only
        and not str(merged.get("cst_profile") or "").strip()
        and merged.get("libcst_usable") is not True
    )
    comments = merged.get("comments_preserved") is True
    source_maps = merged.get("source_map_preserved") is True
    reason = ""
    if claimed and libcst:
        reason = LIBCST_NOT_USABLE
    elif claimed and ast_only:
        reason = AST_ONLY_NOT_CST
    elif claimed and undeclared:
        reason = UNDECLARED_CST_TRANSFORM
    elif claimed and preserving and not comments:
        reason = COMMENTS_NOT_PRESERVED
    elif claimed and preserving and not source_maps:
        reason = SOURCE_MAP_NOT_PRESERVED
    elif claimed and preserving and comments and source_maps:
        reason = CST_PRESERVED
    elif claimed:
        reason = UNDECLARED_CST_TRANSFORM
    blocks = bool(
        claimed
        and reason
        in {
            LIBCST_NOT_USABLE,
            AST_ONLY_NOT_CST,
            UNDECLARED_CST_TRANSFORM,
            COMMENTS_NOT_PRESERVED,
            SOURCE_MAP_NOT_PRESERVED,
        }
    )
    return {
        "accepted_as_authority": False,
        "completes_task": False,
        "claimed": claimed,
        "preserved": reason == CST_PRESERVED,
        "blocks_completion": blocks,
        "reason_code": reason,
    }


def cst_codemod_blocks_completion(state: Mapping[str, Any] | None) -> bool:
    return bool(cst_codemod_view(state)["blocks_completion"])


__all__ = [
    "AST_ONLY_NOT_CST",
    "COMMENTS_NOT_PRESERVED",
    "CST_PRESERVED",
    "LIBCST_NOT_USABLE",
    "SOURCE_MAP_NOT_PRESERVED",
    "UNDECLARED_CST_TRANSFORM",
    "claims_cst_codemod",
    "cst_codemod_blocks_completion",
    "cst_codemod_view",
]
