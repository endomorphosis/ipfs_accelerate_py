"""Closed measurements for the explicitly selected capsule-reference route.

No provider operations or file access occur here. The router performs admission
and an authenticated first-page probe before calling this exporter. That probe
does not establish complete capsule retrieval or future source freshness.
"""
from __future__ import annotations

import hashlib
import json
import re

from .semantic_working_set import SemanticWorkingSetSelection, validate_working_set_selection_receipt
from .semantic_router_translation import (
    EncodedSemanticPrompt, census_semantic_router_prompt, DETAILED_COMPONENT_CENSUS_SCHEMA,
)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _digest(value):
    return None if value is None else _sha(_json(value))


def _require_sha256(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("working-set census requires a closed SHA-256 field")


def working_set_census_receipt(*, selection, encoded, prompt, model, workspace, provider,
        components, source_receipt, preflight, advisory, boundary_sha256, reply,
        formalization, doctor, public):
    """Export fixed labels, counts and hashes; retain identifiers only as hashes."""
    if type(provider) is not str or provider not in {"codex_cli", "grok_cli"}:
        raise ValueError("working-set census requires a supported provider label")
    _require_sha256(boundary_sha256)
    # Only this fixed native schema is currently supported. Validate the two
    # fields copied into the closed export, rather than trusting arbitrary
    # strings or counts supplied to this otherwise pure helper.
    if reply is not None:
        from .coding_reply_contract import coding_completion_schema
        schema = _json(coding_completion_schema())
        if (type(reply) is not dict
                or reply.get("native_output_schema_sha256") != _sha(schema)
                or type(reply.get("native_output_schema_bytes")) is not int
                or reply["native_output_schema_bytes"] != len(schema.encode("utf-8"))):
            raise ValueError("working-set census native reply schema differs")
    if (type(selection) is not SemanticWorkingSetSelection or selection.dispatch_usable is not True
            or selection.historical_replay is not False):
        raise ValueError("current working-set selection required for census")
    if type(encoded) is not EncodedSemanticPrompt or encoded.freshness_checked is not True:
        raise ValueError("current semantic encoding required for working-set census")
    chosen = validate_working_set_selection_receipt(selection.receipt)
    if (chosen["selected_mode"] != selection.selected_mode
            or chosen["selected_complete_sha256"] != _sha(selection.provider_prompt)
            or chosen["selected_complete_bytes"] != len(selection.provider_prompt.encode())):
        raise ValueError("working-set census selected input differs")
    # Equality to the closed producer also rejects caller-supplied extra labels
    # or contents. The components remain an overlapping view of the same input.
    if _json(components) != _json(census_semantic_router_prompt(encoded=encoded,
            census_schema=DETAILED_COMPONENT_CENSUS_SCHEMA)):
        raise ValueError("working-set census components differ")
    if (encoded.native_prompt != prompt or components["native_prompt_sha256"] != _sha(prompt)
            or components["native_prompt_bytes"] != len(prompt.encode())):
        raise ValueError("working-set census original input differs")
    view = chosen["view_receipt"]
    if (view["native_transport_sha256"] != components["provider_prompt_sha256"]
            or view["native_transport_bytes"] != components["provider_prompt_bytes"]
            or source_receipt["descriptor_sha256"] != view["binding"]["descriptor_sha256"]
            or source_receipt["session_id"] != view["binding"]["descriptor_session_id"]):
        raise ValueError("working-set census source binding differs")
    if (preflight is not None) is not chosen["selected"]:
        raise ValueError("working-set census preflight differs from selection")
    probe = None
    if preflight is not None:
        fields = {"schema", "requests", "session_id", "capsule_index", "capsule_sha256",
            "response_bytes", "response_sha256", "provider_input_bytes",
            "source_freshness_checked", "proof_authority"}
        if (type(preflight) is not dict or set(preflight) != fields
                or preflight["schema"] != "supervisor-semantic-working-set-preflight@1"
                or type(preflight["requests"]) is not int or preflight["requests"] != 1
                or type(preflight["capsule_index"]) is not int or preflight["capsule_index"] != 0
                or type(preflight["response_bytes"]) is not int or not 0 < preflight["response_bytes"] <= 4096
                or type(preflight["provider_input_bytes"]) is not int or preflight["provider_input_bytes"] != 0
                or preflight["session_id"] != source_receipt["session_id"]
                or preflight["source_freshness_checked"] is not True or preflight["proof_authority"] is not False):
            raise ValueError("working-set census requires the bounded source preflight")
        _require_sha256(preflight["response_sha256"])
        _require_sha256(preflight["capsule_sha256"])
        probe = dict(schema="supervisor-semantic-working-set-census-preflight@1",
            requests=1, capsules_probed=1, response_bytes=preflight["response_bytes"],
            response_sha256=preflight["response_sha256"], capsule_sha256=preflight["capsule_sha256"],
            session_id_sha256=_sha(preflight["session_id"]), provider_input_bytes=0,
            source_freshness_checked=True, complete_capsule_expansion_verified=False, proof_authority=False)
    measurements = dict(schema="supervisor-semantic-working-set-census@1",
        **{key: chosen[key] for key in ("requested_profile", "selected_mode", "fallback_reason", "selection_basis",
            *(name + suffix for name in ("native_complete", "candidate_complete", "selected_complete")
              for suffix in ("_sha256", "_bytes")))},
        **{key: view[key] for key in ("capsule_count", "capsule_native_bytes", "deferred_field_count",
            "deferred_field_allowlist", "native_transport_bytes", "native_transport_sha256",
            "candidate_view_bytes", "candidate_view_sha256")},
        selection_receipt_sha256=_digest(chosen), projection_receipt_sha256=_digest(view),
        descriptor_sha256=view["binding"]["descriptor_sha256"],
        descriptor_session_id_sha256=_sha(view["binding"]["descriptor_session_id"]),
        semantic_binding_sha256=_digest(view["binding"]["semantic_binding"]),
        selected_complete_byte_reduction=chosen["native_complete_bytes"] - chosen["selected_complete_bytes"],
        candidate_complete_byte_reduction=chosen["native_complete_bytes"] - chosen["candidate_complete_bytes"],
        behavioral_fields_deferred=False, raw_source_deferred=False, complete_expansion_verified=False,
        provider_token_usage=None, token_estimate=False, complete_run_token_savings_claimed=False)
    result = dict(schema="router-semantic-working-set-census@1", purpose="coding", observation_only=True,
        provider=provider, model_sha256=_sha(model), workspace_identity_sha256=_sha(str(workspace)),
        native_prompt_sha256=_sha(prompt), native_prompt_bytes=len(prompt.encode()),
        container_boundary_artifact_sha256=boundary_sha256,
        source_view_receipt_sha256=_digest(source_receipt),
        workspace_advisory_sha256=_sha(advisory), workspace_advisory_bytes=len(advisory.encode()),
        selected_reply_contract_sha256=_digest(reply), source_formalization_receipt_sha256=_digest(formalization),
        doctor_residual_included=doctor is not None, doctor_residual_receipt_sha256=_digest(doctor),
        public_instruction_included=public is not None, public_instruction_receipt_sha256=_digest(public),
        repository_common_dir_checked=True, current_semantic_encoder_used=True,
        container_boundary_checked=True, credential_isolation_checked=True,
        input_measurement_scope="router-model-prompt-with-literal-wrappers-and-reply-instruction@1",
        cli_system_prompt_measured=False, native_output_schema_tokens_measured=False,
        native_output_schema_sha256=(reply or {}).get("native_output_schema_sha256"),
        native_output_schema_bytes=(reply or {}).get("native_output_schema_bytes"),
        semantic_prompt_components=components, census=measurements, semantic_working_set_preflight=probe,
        provider_calls=0, router_calls=0, provider_dispatch_observed=False, planning_performed_by_census=False,
        provider_token_usage=None, total_token_savings_measured=False, raw_content_exported=False,
        source_freshness_authority=False, proof_authority=False, execution_authority=False,
        completion_authority=False, settlement_authority=False, publication_authority=False,
        required_fact_omission_authority=False)
    if len(_json(result).encode()) > 32768:
        raise ValueError("working-set census exceeds its closed output bound")
    return result
