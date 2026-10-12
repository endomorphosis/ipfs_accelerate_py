"""Explicit query-reference presentation; legacy coding input stays unchanged."""
from __future__ import annotations

POLICIES = ("legacy", "required-instruction-ref@1")


def validate_policy(value: str, *, purpose: str = "coding", provider: str = "codex_cli",
                    semantic_repository=True, semantic_metadata_view: str = "legacy",
                    source_input_policy: str = "legacy", census_only: bool = False) -> str:
    if type(value) is not str or value not in POLICIES:
        raise ValueError("explicit supported query reference policy required")
    if value != "legacy":
        if purpose != "coding" or provider != "codex_cli" or semantic_repository is None or census_only:
            raise ValueError("query reference policy requires an actual full indexed Codex coding route")
        if semantic_metadata_view != "legacy" or source_input_policy != "legacy":
            raise ValueError("query reference v1 excludes metadata and source input selection")
    return value


def validate_bound_receipt(value: str, receipt: dict, *, semantic_metadata_view: str = "legacy",
                           source_input_policy: str = "legacy") -> dict:
    from .semantic_query_reference import validate_semantic_query_reference_receipt
    validate_policy(value, semantic_metadata_view=semantic_metadata_view, source_input_policy=source_input_policy)
    if value == "legacy":
        raise ValueError("legacy input has no query reference projection")
    validate_semantic_query_reference_receipt(receipt)
    if type(receipt.get("reference_count")) is not int or receipt["reference_count"] != 1:
        raise ValueError("query reference receipt requires one exact inline instruction reference")
    return receipt
