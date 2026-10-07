"""Closed observations over an upstream-verified readable metadata view.

This helper reads supplied strings only. It does not query a database, inspect
files, run a provider, verify current sources or confer dispatch authority.
Every identity exported here is a SHA256 digest; text, rows and binding values
remain in the existing owner-held inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re

from ..context.context_contracts import canonical_context_json_bytes
from .semantic_metadata_view import (
    GROUP_FIELDS, MAX_ROWS, TOKEN_PROXY, TRANSPORT_SCHEMA, VIEW_SCHEMA,
    SemanticMetadataView, SemanticMetadataViewError,
    project_semantic_metadata_view, restore_semantic_metadata_view,
    select_semantic_metadata_view,
)


CENSUS_SCHEMA = "supervisor-semantic-metadata-census@1"
FIXED_MAX_INPUT_BYTES = 256_000
MAX_ID_BYTES = 4096
_FALSE_FIELDS = (
    "source_freshness_verified", "program_semantics_proved", "proof_authority",
    "omission_authority", "execution_authority", "dispatch_authority",
    "completion_authority", "publication_authority",
)
_COPY_DIGEST_FIELDS = (
    "source_manifest_sha256", "native_core_sha256", "native_context_sha256",
    "native_suffix_sha256", "original_translated_semantic_sha256",
    "native_transport_sha256", "candidate_view_sha256", "common_bindings_sha256",
)
_COMPLETE_OBSERVATION_FIELDS = (
    "native_complete_sha256", "native_complete_bytes", "native_complete_proxy_tokens",
    "candidate_complete_sha256", "candidate_complete_bytes", "candidate_complete_proxy_tokens",
    "selected_complete_sha256", "selected_complete_bytes", "selected_complete_proxy_tokens",
)


class SemanticMetadataCensusError(ValueError):
    """A closed input/shape check failed; messages never include source text."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(value) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > MAX_ID_BYTES:
        raise SemanticMetadataCensusError("invalid bounded census identity")
    return value


def _digest(value) -> str:
    if type(value) is not str or re.fullmatch("[0-9a-f]{64}", value) is None:
        raise SemanticMetadataCensusError("invalid census digest")
    return value


def _source_identity(value, aliases: Mapping) -> str:
    if type(value) is str:
        return _identity(value)
    if (type(value) is dict and set(value) == {"$semantic_ref"}
            and type(value["$semantic_ref"]) is str and value["$semantic_ref"] in aliases):
        return _identity(aliases[value["$semantic_ref"]])
    raise SemanticMetadataCensusError("invalid census source reference")


def _source_counts(native: dict) -> dict:
    semantic = native["translated_semantic"]
    manifest = semantic["manifest"]
    aliases = native["translation_table"]
    if (type(manifest) is not dict or not 1 <= len(manifest) <= MAX_ROWS
            or type(aliases) is not dict):
        raise SemanticMetadataCensusError("invalid bounded census source population")
    captured = set()
    for name, binding in manifest.items():
        _identity(name)  # Shape check only: names are never exported.
        if type(binding) is not dict:
            raise SemanticMetadataCensusError("invalid census source binding")
        captured.add(_identity(binding.get("source_cid")))
        _digest(binding.get("sha256"))
    capsule_sources = set()
    admission_sources = set()
    for row in semantic["capsules"]:
        capsule_sources.add(_source_identity(row.get("source_cid"), aliases))
    for row in semantic["admissions"]:
        admission_sources.add(_source_identity(row["ref"].get("source_cid"), aliases))
    if not capsule_sources <= captured or not admission_sources <= captured:
        raise SemanticMetadataCensusError("census source groups escape captured population")
    return {
        "captured_source_count": len(manifest),
        "captured_source_group_count": len(captured),
        "capsule_source_group_count": len(capsule_sources),
        "admission_source_group_count": len(admission_sources),
    }


def census_semantic_metadata_view(
    *, view: SemanticMetadataView, native_complete_prompt: str,
    candidate_complete_prompt: str, fixed_max_input_bytes: int = FIXED_MAX_INPUT_BYTES,
) -> dict:
    """Return closed counts/digests after exact restoration and input selection.

    The owner must supply its already verified view and complete prompt strings.
    Representation checks do not establish that owner's custody or freshness.
    Matched-input eligibility is a bounded representation observation only.
    """
    if type(view) is not SemanticMetadataView:
        raise SemanticMetadataCensusError("typed semantic metadata view required")
    if type(fixed_max_input_bytes) is not int or fixed_max_input_bytes != FIXED_MAX_INPUT_BYTES:
        raise SemanticMetadataCensusError("fixed census complete-input bound required")
    try:
        restored = restore_semantic_metadata_view(provider_prompt=view.provider_prompt, receipt=view.receipt)
        if restored != view.native_prompt:
            raise SemanticMetadataCensusError("census exact restoration differs")
        # Require the fixed upstream projector's actual candidate, rather than
        # a caller-selected alternate representation with a self-rehashed receipt.
        if project_semantic_metadata_view(restored) != view:
            raise SemanticMetadataCensusError("census projected candidate differs")
        selection = select_semantic_metadata_view(
            view=view, native_complete_prompt=native_complete_prompt,
            candidate_complete_prompt=candidate_complete_prompt,
        )
        native = json.loads(restored)  # The existing restore already checked canonical bounded JSON.
        candidate = json.loads(view.provider_prompt)
        receipt, chosen = view.receipt, selection.receipt
        source_counts = _source_counts(native)
        common = candidate["common_bindings"]
        shared_fields = {}
        for group in sorted(GROUP_FIELDS):
            values = common[group]
            if type(values) is not dict or not set(values) <= GROUP_FIELDS[group]:
                raise SemanticMetadataCensusError("census common field population differs")
            shared_fields[group] = sorted(values)
        # Export a fixed construction, never either upstream receipt wholesale.
        output = {
            "schema": CENSUS_SCHEMA, "source_transport_schema": TRANSPORT_SCHEMA,
            "view_schema": VIEW_SCHEMA, "token_proxy": TOKEN_PROXY,
            "fixed_max_input_bytes": FIXED_MAX_INPUT_BYTES,
            "task_id_sha256": _sha(_identity(receipt["task_id"])),
            "translation_cid_sha256": _sha(_identity(receipt["translation_cid"])),
            "scope_cid_sha256": _sha(_identity(receipt["scope_cid"])),
            "semantic_root_cid_sha256": _sha(_identity(receipt["semantic_root_cid"])),
            **{field: _digest(receipt[field]) for field in _COPY_DIGEST_FIELDS},
            "view_receipt_sha256": _sha(view.receipt_json),
            "selection_receipt_sha256": _sha(selection.receipt_json),
            "capsule_count": receipt["capsule_count"], "admission_count": receipt["admission_count"],
            **source_counts, "shared_field_names": shared_fields,
            "shared_field_counts": {group: len(fields) for group, fields in shared_fields.items()},
            "shared_field_count": sum(len(fields) for fields in shared_fields.values()),
            "exact_restoration_verified": True,
            "selected_mode": selection.selected_mode,
            "fallback_reason": chosen["fallback_reason"],
            **{field: chosen[field] for field in _COMPLETE_OBSERVATION_FIELDS},
            "native_complete_within_bound": chosen["native_complete_bytes"] <= FIXED_MAX_INPUT_BYTES,
            "candidate_complete_within_bound": chosen["candidate_complete_bytes"] <= FIXED_MAX_INPUT_BYTES,
            "selected_complete_within_bound": chosen["selected_complete_bytes"] <= FIXED_MAX_INPUT_BYTES,
            "eligible_for_matched_input_comparison": (
                chosen["native_complete_bytes"] <= FIXED_MAX_INPUT_BYTES
                and chosen["candidate_complete_bytes"] <= FIXED_MAX_INPUT_BYTES
                and selection.selected_mode == "common-bindings@1"
            ),
            "candidate_complete_byte_reduction": chosen["native_complete_bytes"] - chosen["candidate_complete_bytes"],
            "selected_complete_byte_reduction": chosen["native_complete_bytes"] - chosen["selected_complete_bytes"],
            "candidate_complete_proxy_reduction": chosen["native_complete_proxy_tokens"] - chosen["candidate_complete_proxy_tokens"],
            "selected_complete_proxy_reduction": chosen["native_complete_proxy_tokens"] - chosen["selected_complete_proxy_tokens"],
            "representation_only": True, "candidate_only": True,
            "actual_provider_tokens_measured": False, "total_token_savings_qualified": False,
            "provider_calls": 0, **{field: False for field in _FALSE_FIELDS},
        }
        if len(canonical_context_json_bytes(output)) > 16_384:
            raise SemanticMetadataCensusError("closed census output exceeds bound")
        return output
    except SemanticMetadataCensusError:
        raise
    except (SemanticMetadataViewError, TypeError, ValueError, KeyError, UnicodeError, RecursionError) as error:
        raise SemanticMetadataCensusError("census input verification failed") from error
