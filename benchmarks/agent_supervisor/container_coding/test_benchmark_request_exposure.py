"""Receipt-only exposure observations; no provider calls or source ingestion."""

from copy import deepcopy
import json

import pytest

from benchmarks.agent_supervisor.container_coding.benchmark_request_exposure import (
    MAX_RENDERED_PROMPT_BYTES,
    request_exposure_observation,
)


def _raw(size=80_000, digest="a" * 64):
    return {"model_prompt_sha256": digest, "model_prompt_bytes": size}


def _metadata(size=80_000, digest="a" * 64):
    # Authored shape of the existing validated comparator projection. This
    # helper does not independently authenticate raw metadata receipts.
    return {
        "status": "reported",
        "measurement_scope": "complete-rendered-prompt-text@1",
        "selected_complete_sha256": digest,
        "selected_complete_bytes": size,
        "selection_basis": "token_counter",
        "token_counter_status": "counted",
        "selected_complete_tokens": 19_001,
        "token_counter_identity": {
            "qualification": "estimate",
            "measurement_scope": "complete-rendered-prompt-text@1",
            "identity_caller_supplied": True,
            "provider_usage_verified": False,
        },
    }


def _unknown_provider(observed):
    for key in ("provider_request_bytes", "provider_request_tokens", "provider_framing_tokens",
                "conversation_history_tokens", "provider_dispatch_verified",
                "provider_ingestion_verified", "provider_cache_hit_verified", "retained_parent_verified"):
        assert observed[key] is None
    assert observed["provider_usage_measured"] is False
    assert observed["token_savings_established"] is False
    assert observed["prompt_content_rehashed"] is False


@pytest.mark.parametrize("raw", [None, [], "private prompt", {}, {"model_prompt_bytes": 80_000},
                                 {"model_prompt_sha256": "a" * 64}])
def test_missing_pair_remains_wholly_unknown(raw):
    observed = request_exposure_observation(raw, semantic_metadata=_metadata())
    assert observed["status"] == "unavailable"
    assert observed["model_prompt_sha256"] is observed["model_prompt_bytes"] is None
    assert observed["selected_metadata_binding"] == "unavailable"
    assert observed["local_prompt_count"] is None
    _unknown_provider(observed)


@pytest.mark.parametrize("field,value", [
    ("model_prompt_sha256", True), ("model_prompt_sha256", ["a" * 64]),
    ("model_prompt_sha256", "A" * 64), ("model_prompt_sha256", "a" * 63),
    ("model_prompt_sha256", "/private/prompt"), ("model_prompt_sha256", "a" * 64 + "\n"),
    ("model_prompt_bytes", True), ("model_prompt_bytes", False),
    ("model_prompt_bytes", 0), ("model_prompt_bytes", -1),
    ("model_prompt_bytes", 80_000.0), ("model_prompt_bytes", "80000"),
    ("model_prompt_bytes", MAX_RENDERED_PROMPT_BYTES + 1),
    ("model_prompt_bytes", 2**100),
])
def test_malformed_pair_does_not_export_partial_commitment(field, value):
    raw = _raw()
    raw[field] = value
    observed = request_exposure_observation(raw)
    assert observed["status"] == "unavailable"
    assert observed["model_prompt_sha256"] is observed["model_prompt_bytes"] is None


@pytest.mark.parametrize("size", [1, MAX_RENDERED_PROMPT_BYTES])
def test_inclusive_supported_bounds_are_reported_without_ingestion_claim(size):
    observed = request_exposure_observation(_raw(size))
    assert observed["status"] == "reported"
    assert observed["model_prompt_bytes"] == size
    assert observed["scope"] == "rendered_user_prompt_only"
    assert observed["selected_metadata_binding"] == "unavailable"
    _unknown_provider(observed)


@pytest.mark.parametrize("field,value", [
    ("selected_complete_sha256", "b" * 64), ("selected_complete_bytes", 79_999),
])
def test_metadata_requires_both_selected_hash_and_byte_count(field, value):
    metadata = _metadata()
    metadata[field] = value
    observed = request_exposure_observation(_raw(), semantic_metadata=metadata)
    assert observed["selected_metadata_binding"] == "mismatched"
    assert observed["model_prompt_bytes"] == 80_000
    assert observed["local_prompt_count"] is None
    _unknown_provider(observed)


@pytest.mark.parametrize("metadata", [None, [], {}, {"status": "unavailable"}])
def test_absent_or_unavailable_metadata_keeps_valid_complete_prompt(metadata):
    observed = request_exposure_observation(_raw(), semantic_metadata=metadata)
    assert observed["status"] == "reported"
    assert observed["selected_metadata_binding"] == "unavailable"
    assert observed["model_prompt_bytes"] == 80_000


@pytest.mark.parametrize("field,value", [
    ("selected_complete_sha256", True), ("selected_complete_sha256", "/private/source"),
    ("selected_complete_bytes", True), ("selected_complete_bytes", 0),
    ("selected_complete_bytes", MAX_RENDERED_PROMPT_BYTES + 1),
    ("measurement_scope", "partial-subtree"), ("status", "unavailable"),
])
def test_malformed_metadata_commitment_cannot_bind(field, value):
    metadata = _metadata()
    metadata[field] = value
    observed = request_exposure_observation(_raw(), semantic_metadata=metadata)
    assert observed["selected_metadata_binding"] == "unavailable"
    assert observed["local_prompt_count"] is None


@pytest.mark.parametrize("qualification", ["estimate", "provider_tokenizer"])
def test_matched_local_count_remains_separate_from_native_usage(qualification):
    metadata = _metadata()
    metadata["token_counter_identity"]["qualification"] = qualification
    observed = request_exposure_observation(_raw(), semantic_metadata=metadata)
    assert observed["selected_metadata_binding"] == "matched"
    assert observed["local_prompt_count"]["tokens"] == 19_001
    assert observed["local_prompt_count"]["qualification"] == qualification
    assert observed["local_prompt_count"]["provider_usage_measured"] is False
    assert observed["local_prompt_count"]["provider_tokenizer_equivalence_verified"] is False
    _unknown_provider(observed)


@pytest.mark.parametrize("field,value", [
    ("selected_complete_tokens", True), ("selected_complete_tokens", -1),
    ("selected_complete_tokens", 2**63), ("token_counter_status", "failed"),
    ("selection_basis", "byte_proxy"), ("token_counter_identity", None),
])
def test_unavailable_local_count_does_not_invalidate_matching_byte_commitment(field, value):
    metadata = _metadata()
    metadata[field] = value
    observed = request_exposure_observation(_raw(), semantic_metadata=metadata)
    assert observed["selected_metadata_binding"] == "matched"
    assert observed["local_prompt_count"] is None


def test_tiny_internal_delta_does_not_replace_full_rendered_prompt_exposure():
    raw = {**_raw(), "semantic_translation": {"provider_prompt_bytes": 20},
           "router_prompt_bytes": 30, "native_prompt_bytes": 80_004}
    metadata = {**_metadata(), "native_complete_bytes": 80_004,
                "candidate_complete_bytes": 80_000}
    observed = request_exposure_observation(raw, semantic_metadata=metadata)
    assert observed["model_prompt_bytes"] == 80_000
    assert observed["model_prompt_bytes"] not in (4, 20, 30)
    assert observed["selected_metadata_binding"] == "matched"
    _unknown_provider(observed)


def test_equal_hashes_do_not_cache_counts_or_infer_provider_cache_hits():
    first_metadata = _metadata()
    second_metadata = deepcopy(first_metadata)
    second_metadata["selected_complete_tokens"] = 20_001
    first = request_exposure_observation(_raw(), semantic_metadata=first_metadata)
    second = request_exposure_observation(_raw(), semantic_metadata=second_metadata)
    assert first["local_prompt_count"]["tokens"] == 19_001
    assert second["local_prompt_count"]["tokens"] == 20_001
    assert first["model_prompt_bytes"] + second["model_prompt_bytes"] == 160_000
    _unknown_provider(first)
    _unknown_provider(second)


def test_projection_does_not_export_private_fields_or_mutate_inputs():
    marker = "PRIVATE_PROMPT_PATH_OR_EXCEPTION"
    raw = {**_raw(), "prompt": marker, "workspace": marker, "error": marker,
           "repository_world_packet": {"retained_parent_verified": True, "body": marker}}
    metadata = {**_metadata(), "prompt": marker, "counter_error": marker}
    before = deepcopy((raw, metadata))
    observed = request_exposure_observation(raw, semantic_metadata=metadata)
    assert (raw, metadata) == before
    assert marker not in json.dumps(observed)
    observed["local_prompt_count"]["tokens"] = 1
    assert metadata["selected_complete_tokens"] == 19_001
    _unknown_provider(observed)
