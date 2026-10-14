"""Explicit source presentation experiment; legacy input stays byte-identical."""
from __future__ import annotations

POLICIES = ("legacy", "source-input-raw@1", "source-input-compact@1",
            "source-input-ast-ref-raw@1", "source-input-ast-ref-compact@1")


def validate_policy(value: str, *, purpose: str = "coding", provider: str = "codex_cli",
                    semantic_repository=True, semantic_metadata_view: str = "legacy",
                    census_only: bool = False, prompt_token_counter=None,
                    prompt_tokenizer_artifact=None, max_input_tokens=None) -> str:
    if type(value) is not str or value not in POLICIES:
        raise ValueError("explicit supported source input policy required")
    if value != "legacy":
        if (purpose != "coding" or provider != "codex_cli" or semantic_repository is None
                or census_only):
            raise ValueError("source input policy requires an actual full indexed Codex coding route")
        if "-ast-ref-" in value and (semantic_metadata_view != "legacy"
                or prompt_token_counter is not None or prompt_tokenizer_artifact is not None
                or max_input_tokens is not None):
            raise ValueError("source input AST reference v1 excludes metadata and tokenizer selection")
    return value


def projection_options(value: str) -> dict:
    validate_policy(value)
    if value == "legacy":
        raise ValueError("legacy input has no source projection")
    return {"format": "compact" if "compact" in value else "raw", "factor_ast": "-ast-ref-" in value}


def validate_bound_receipt(value: str, receipt: dict, *, semantic_metadata_view: str = "legacy") -> dict:
    """Qualify actual positive input inclusion, separately from optional helper use."""
    from .source_input_view import validate_source_input_view_receipt
    validate_policy(value, semantic_metadata_view=semantic_metadata_view)
    options = projection_options(value)
    validate_source_input_view_receipt(receipt)
    if (receipt['format'] != options['format'] or receipt['factor_ast'] is not options['factor_ast']
            or receipt['freshness_checked'] is not True or receipt['workspace_checked'] is not True
            or receipt['views_included'] < 1 or receipt['block_bytes'] < 1
            or options['format'] == 'compact' and receipt['compact_views_included'] < 1):
        raise ValueError('source input receipt does not establish the requested presentation')
    return receipt
