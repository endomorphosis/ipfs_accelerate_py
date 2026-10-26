"""Bound reported rendered-prompt exposure without reading prompt contents.

The input is an invocation receipt, not a provider request or transcript. The
optional metadata argument is the comparator's already validated
``_semantic_metadata_observation`` projection. Matching its selected complete
prompt commitment is an internal receipt join, not evidence of dispatch,
provider ingestion, retained-parent custody, caching, or native token usage.
"""

from __future__ import annotations

import re


SCHEMA = "benchmark-request-exposure-observation@1"
# Matches runtime/local_prompt_tokenizer.py:MAX_PROMPT_BYTES. Values outside
# this diagnostic's supported scope remain unknown; no truncation or recount.
MAX_RENDERED_PROMPT_BYTES = 256_000
_COMPLETE_PROMPT_SCOPE = "complete-rendered-prompt-text@1"


def _pair(value, prefix):
    if type(value) is not dict:
        return None
    digest, size = value.get(prefix + "sha256"), value.get(prefix + "bytes")
    if (type(digest) is not str or len(digest) != 64
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or type(size) is not int or not 1 <= size <= MAX_RENDERED_PROMPT_BYTES):
        return None
    return digest, size


def _local_count(metadata):
    """Project only a bound local count; never promote its qualification."""
    identity = metadata.get("token_counter_identity")
    count = metadata.get("selected_complete_tokens")
    if (metadata.get("selection_basis") != "token_counter"
            or metadata.get("token_counter_status") != "counted"
            or type(count) is not int or not 1 <= count <= 2**63 - 1
            or type(identity) is not dict
            or type(identity.get("qualification")) is not str
            or identity["qualification"] not in {"estimate", "provider_tokenizer"}
            or identity.get("measurement_scope") != _COMPLETE_PROMPT_SCOPE
            or identity.get("identity_caller_supplied") is not True
            or identity.get("provider_usage_verified") is not False):
        return None
    return {
        "tokens": count,
        "qualification": identity["qualification"],
        "scope": "rendered_user_prompt_only",
        "provenance": "validated_semantic_metadata_projection",
        "provider_usage_measured": False,
        "provider_tokenizer_equivalence_verified": False,
    }


def request_exposure_observation(raw, *, semantic_metadata=None):
    """Project bounded commitments for one receipt, with no cross-call cache.

    Use the complete ``model_prompt`` pair, never a smaller translated subtree
    or a representation delta. A malformed/missing pair stays wholly unknown.
    Identical hashes in different receipts do not establish provider cache hits
    and do not remove either invocation from an exposure population.
    """
    pair = _pair(raw, "model_prompt_")
    metadata_pair = None
    if (type(semantic_metadata) is dict
            and semantic_metadata.get("status") == "reported"
            and semantic_metadata.get("measurement_scope") == _COMPLETE_PROMPT_SCOPE):
        metadata_pair = _pair(semantic_metadata, "selected_complete_")
    binding = ("unavailable" if pair is None or metadata_pair is None
               else "matched" if pair == metadata_pair else "mismatched")
    return {
        "schema": SCHEMA,
        "status": "reported" if pair is not None else "unavailable",
        "scope": "rendered_user_prompt_only",
        "model_prompt_sha256": pair[0] if pair is not None else None,
        "model_prompt_bytes": pair[1] if pair is not None else None,
        "max_rendered_prompt_bytes": MAX_RENDERED_PROMPT_BYTES,
        "selected_metadata_binding": binding,
        "local_prompt_count": _local_count(semantic_metadata) if binding == "matched" else None,
        "provider_request_bytes": None,
        "provider_request_tokens": None,
        "provider_framing_tokens": None,
        "conversation_history_tokens": None,
        "provider_dispatch_verified": None,
        "provider_ingestion_verified": None,
        "provider_cache_hit_verified": None,
        "retained_parent_verified": None,
        "prompt_content_rehashed": False,
        "provider_usage_measured": False,
        "token_savings_established": False,
    }
