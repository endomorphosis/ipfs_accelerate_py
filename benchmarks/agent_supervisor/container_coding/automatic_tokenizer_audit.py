"""Validate recorded automatic counters without loading assets or recounting.

Only the closed single-comparison prompt-selection v2 contract is supported.
Multi-candidate codec replay remains outside the historical auditor's scope.
Estimates are checked as estimates throughout; they never become token usage.
"""
from __future__ import annotations

import re

SELECTION_SCHEMA = "supervisor-model-prompt-selection@2"
SOURCE_SCHEMA = "supervisor-automatic-prompt-tokenizer@1"
HISTORICAL_SCHEMA = "terminal-historical-automatic-prompt-tokenizer-provenance@1"
_BINDING_FIELDS = {"provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy"}
_SELECTION_FIELDS = {
    "schema", "selected_representation", "selection_reason", "candidates", "measurement_status",
    "measurement_error_type", "tokenizer_binding", "count_scope", "selected_content_tokens",
    "max_input_tokens", "max_prompt_bytes", "total_session_token_savings", "completion_authority",
    "token_count_kind", "tokenizer_compatibility_basis", "selected_estimated_content_tokens",
    "token_budget_basis", "token_budget_guaranteed",
}
_SOURCE_FIELDS = {
    "schema", "backend", "backend_version", "encoding", "encoding_spec_revision", "rank_sha256",
    "count_kind", "compatibility_basis", "reason", "model_mapping_match", "model_mapping_prefix",
    "offline", "remote_tokenizer_compatibility_verified", "provider_usage", "completion_authority",
}
_RANK_HASHES = {
    "r50k_base": "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
    "p50k_base": "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
    "cl100k_base": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    "o200k_base": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
}
_RANK_HASHES["p50k_edit"] = _RANK_HASHES["p50k_base"]
_RANK_HASHES["o200k_harmony"] = _RANK_HASHES["o200k_base"]
# Historical mapping snapshot from the resolver's reviewed tiktoken 0.13.0.
# Do not import the current installed package to reinterpret a recorded route.
_EXACT_MODELS = {}
for _encoding, _names in (
    ("o200k_base", "o1 o3 o4-mini gpt-5 gpt-4.1 gpt-4o"),
    ("cl100k_base", "gpt-4 gpt-3.5-turbo gpt-3.5 gpt-35-turbo davinci-002 babbage-002 "
     "text-embedding-ada-002 text-embedding-3-small text-embedding-3-large"),
    ("p50k_base", "text-davinci-003 text-davinci-002 code-davinci-002 code-davinci-001 "
     "code-cushman-002 code-cushman-001 davinci-codex cushman-codex"),
    ("p50k_edit", "text-davinci-edit-001 code-davinci-edit-001"),
    ("r50k_base", "text-davinci-001 text-curie-001 text-babbage-001 text-ada-001 davinci curie babbage ada "
     "text-similarity-davinci-001 text-similarity-curie-001 text-similarity-babbage-001 text-similarity-ada-001 "
     "text-search-davinci-doc-001 text-search-curie-doc-001 text-search-babbage-doc-001 text-search-ada-doc-001 "
     "code-search-babbage-code-001 code-search-ada-code-001"),
    ("gpt2", "gpt2 gpt-2"),
):
    _EXACT_MODELS.update(dict.fromkeys(_names.split(), _encoding))
_PREFIX_MODELS = {
    "o1-": "o200k_base", "o3-": "o200k_base", "o4-mini-": "o200k_base", "gpt-5-": "o200k_base",
    "gpt-4.5-": "o200k_base", "gpt-4.1-": "o200k_base", "chatgpt-4o-": "o200k_base", "gpt-4o-": "o200k_base",
    "gpt-4-": "cl100k_base", "gpt-3.5-turbo-": "cl100k_base", "gpt-35-turbo-": "cl100k_base",
    "gpt-oss-": "o200k_harmony", "ft:gpt-4o": "o200k_base", "ft:gpt-4": "cl100k_base",
    "ft:gpt-3.5-turbo": "cl100k_base", "ft:davinci-002": "cl100k_base", "ft:babbage-002": "cl100k_base",
}


def _text(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 512 and value == value.strip()


def _binding_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and value == value.strip()


def _digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _binding(selection: dict, original: dict | None = None) -> dict | None:
    binding = selection.get("tokenizer_binding")
    if binding is not None and (not isinstance(binding, dict) or set(binding) != _BINDING_FIELDS
            or not all(_binding_text(value) for value in binding.values())
            or original is not None and (binding["provider"] != original.get("provider")
                                        or binding["model"] != original.get("model"))):
        raise ValueError("prompt tokenizer differs from recorded provider/model")
    return binding


def validate_prompt_selection(selection: dict, receipt: dict, original: dict) -> None:
    """Validate v2 choice algebra directly, with no fabricated measured receipt."""
    if (not isinstance(selection, dict) or set(selection) != _SELECTION_FIELDS
            or selection.get("schema") != SELECTION_SCHEMA
            or selection["count_scope"] != "rendered_user_prompt_only"
            or selection["total_session_token_savings"] is not None
            or selection["completion_authority"] is not False
            or selection["token_budget_guaranteed"] is not False):
        raise ValueError("invalid or unsupported v2 prompt selection contract")
    selected, status = selection["selected_representation"], selection["measurement_status"]
    kind, basis = selection["token_count_kind"], selection["tokenizer_compatibility_basis"]
    if status not in {"measured", "estimated", "unavailable", "failed"}:
        raise ValueError("invalid prompt measurement status")
    binding = _binding(selection, original)
    if status == "unavailable":
        if binding is not None or kind is not None or basis is not None:
            raise ValueError("unavailable prompt count has a counter binding")
    elif (binding is None or kind not in {"tokenizer", "estimate"} or not _binding_text(basis)
          or status == "measured" and kind != "tokenizer" or status == "estimated" and kind != "estimate"):
        raise ValueError("prompt count quality differs from its measurement status")
    error = selection["measurement_error_type"]
    if (status == "failed" and not _text(error)) or (status != "failed" and error is not None):
        raise ValueError("invalid prompt measurement error")
    expected = {"native", "semantic"} if receipt.get("semantic_encoding_candidate") is not None else {"native"}
    candidates = selection["candidates"]
    if not isinstance(candidates, dict) or set(candidates) != expected or selected not in expected:
        raise ValueError("prompt candidate population differs from verified encoding")
    counted = status in {"measured", "estimated"}
    field = "estimated_input_tokens" if kind == "estimate" else "input_tokens"
    for candidate in candidates.values():
        fields = {"model_prompt_sha256", "model_prompt_bytes", "input_tokens"}
        if kind == "estimate":
            fields.add("estimated_input_tokens")
        if not isinstance(candidate, dict) or set(candidate) != fields or not _digest(candidate["model_prompt_sha256"]):
            raise ValueError("invalid rendered prompt candidate")
        if type(candidate["model_prompt_bytes"]) is not int or candidate["model_prompt_bytes"] < 1:
            raise ValueError("invalid candidate byte count")
        if kind == "estimate" and candidate["input_tokens"] is not None:
            raise ValueError("estimated content cannot be recorded as measured tokens")
        count = candidate[field]
        if (counted and (type(count) is not int or count < 1)) or (not counted and count is not None):
            raise ValueError("invalid candidate content count")
    cap, token_cap = selection["max_prompt_bytes"], selection["max_input_tokens"]
    if type(cap) is not int or not 1 <= cap <= 256_000:
        raise ValueError("prompt byte ceiling exceeds runner policy")
    if token_cap is not None and (type(token_cap) is not int or token_cap < 1 or not counted):
        raise ValueError("invalid complete prompt token ceiling")
    expected_budget = ("estimated_content_tokens" if kind == "estimate" else "local_tokenizer_content_tokens") if token_cap is not None else None
    if selection["token_budget_basis"] != expected_budget:
        raise ValueError("token ceiling basis differs from count quality")
    chosen = candidates[selected]
    recorded = selection["selected_estimated_content_tokens"] if kind == "estimate" else selection["selected_content_tokens"]
    unused = selection["selected_content_tokens"] if kind == "estimate" else selection["selected_estimated_content_tokens"]
    if (unused is not None or recorded != chosen[field]
            or recorded is not None and type(recorded) is not int
            or chosen["model_prompt_sha256"] != receipt["model_prompt_sha256"]
            or chosen["model_prompt_bytes"] != receipt["model_prompt_bytes"]
            or chosen["model_prompt_bytes"] > cap or token_cap is not None and chosen[field] > token_cap):
        raise ValueError("chosen prompt differs from dispatched input or exceeds bounds")
    expected_selected, reason = "native", "semantic_not_requested"
    if "semantic" in candidates:
        reason = "native_not_larger" if counted else "tokenizer_" + status
        if (counted and candidates["semantic"][field] < candidates["native"][field]
                and candidates["semantic"]["model_prompt_bytes"] <= cap):
            expected_selected = "semantic"
            reason = ("semantic_strictly_fewer_estimated_content_tokens" if kind == "estimate"
                      else "semantic_strictly_fewer_content_tokens")
    if selected != expected_selected or selection["selection_reason"] != reason:
        raise ValueError("prompt choice does not follow its recorded count quality")
    source = receipt.get("prompt_tokenizer_source")
    if isinstance(source, dict) and source.get("schema") == "supervisor-local-prompt-tokenizer-provenance@1":
        if kind != "tokenizer" or basis != "owner_declaration":
            raise ValueError("owner tokenizer source differs from selected count quality")


def _recorded_mapping(binding: dict) -> tuple[str | None, str | None, str | None]:
    if binding["provider"] not in {"openai", "codex_cli"}:
        return None, None, None
    model = binding["model"]
    if model in _EXACT_MODELS:
        return _EXACT_MODELS[model], "exact", None
    for prefix, encoding in _PREFIX_MODELS.items():
        if model.startswith(prefix):
            return encoding, "prefix", prefix
    return None, None, None


def historical_tokenizer_source(source: object, selection: object) -> dict:
    """Check recorded automatic provenance; never open a cache or count text."""
    if (not isinstance(source, dict) or set(source) != _SOURCE_FIELDS or source.get("schema") != SOURCE_SCHEMA
            or source["offline"] is not True or source["remote_tokenizer_compatibility_verified"] is not False
            or source["provider_usage"] is not False or source["completion_authority"] is not False
            or not isinstance(selection, dict) or selection.get("schema") != SELECTION_SCHEMA):
        raise ValueError("invalid historical automatic tokenizer provenance")
    binding = _binding(selection)
    if (binding is None or source["count_kind"] != selection.get("token_count_kind")
            or source["compatibility_basis"] != selection.get("tokenizer_compatibility_basis")):
        raise ValueError("automatic tokenizer source differs from selection quality")
    reason, encoding, version = source["reason"], source["encoding"], source["backend_version"]
    mapped, match, prefix = _recorded_mapping(binding)
    if source["backend"] == "utf8-bytes":
        if (version != "1" or encoding != "utf8-div4" or source["encoding_spec_revision"] is not None
                or source["rank_sha256"] is not None or source["count_kind"] != "estimate"
                or source["compatibility_basis"] != "utf8_byte_estimate"
                or source["model_mapping_match"] is not None or source["model_mapping_prefix"] is not None
                or reason not in {"tiktoken_runtime_unavailable", "unsupported_tiktoken_version",
                                  "tiktoken_runtime_version_mismatch", "cached_tokenizer_unavailable"}
                or binding["tokenizer_id"] != "utf8-div4" or binding["tokenizer_revision"] != "utf8-div4@1"
                or binding["special_token_policy"] != "utf8-literal-text@1"):
            raise ValueError("invalid historical UTF-8 estimate binding")
    elif source["backend"] == "tiktoken":
        if (version != "0.13.0" or source["encoding_spec_revision"] != "tiktoken-openai-public@0.13.0"
                or encoding not in _RANK_HASHES or source["rank_sha256"] != _RANK_HASHES[encoding]
                or binding["tokenizer_id"] != "tiktoken/" + encoding
                or binding["tokenizer_revision"] != f"tiktoken/{version}/{encoding}/sha256:{source['rank_sha256']}/ordinary-spec:1"
                or binding["special_token_policy"] != "tiktoken.encode_ordinary:literal-text@1"):
            raise ValueError("invalid historical cached tokenizer binding")
        if source["count_kind"] == "tokenizer":
            if (source["compatibility_basis"] != "installed_tiktoken_model_mapping"
                    or reason != "installed_tiktoken_model_mapping" or mapped != encoding
                    or source["model_mapping_match"] != match or source["model_mapping_prefix"] != prefix):
                raise ValueError("historical model mapping differs from tokenizer provenance")
        elif source["count_kind"] == "estimate":
            if encoding != "o200k_base" or source["compatibility_basis"] != "offline_o200k_base_proxy":
                raise ValueError("invalid historical tokenizer proxy quality")
            expected_match, expected_prefix = match, prefix
            if reason == "provider_mapping_unavailable":
                valid = binding["provider"] not in {"openai", "codex_cli"}
            elif reason == "unmapped_model":
                valid = binding["provider"] in {"openai", "codex_cli"} and mapped is None
            elif reason == "unsupported_mapped_encoding":
                valid = mapped is not None and mapped not in _RANK_HASHES
            elif reason == "mapped_encoding_unavailable":
                valid = mapped in _RANK_HASHES and mapped != "o200k_base"
            elif reason == "installed_model_mapping_unavailable":
                valid = binding["provider"] in {"openai", "codex_cli"}
                expected_match = expected_prefix = None
            else:
                valid = False
            if (not valid or source["model_mapping_match"] != expected_match
                    or source["model_mapping_prefix"] != expected_prefix):
                raise ValueError("historical proxy reason differs from its recorded route")
        else:
            raise ValueError("invalid automatic tokenizer count quality")
    else:
        raise ValueError("unsupported historical automatic tokenizer backend")
    return {
        **source, "schema": HISTORICAL_SCHEMA, "source_schema": SOURCE_SCHEMA,
        "tokenizer_revision": binding["tokenizer_revision"], "recorded_binding_validated": True,
        "source_currentness_revalidated": False, "counts_recomputed": False,
        "provider_tokenizer_compatibility_verified": False,
    }
