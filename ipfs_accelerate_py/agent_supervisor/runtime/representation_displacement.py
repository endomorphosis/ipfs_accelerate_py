"""Closed observations of existing, lossless complete-input selections.

The caller supplies the original typed view and selection. This module neither
selects a new treatment nor calls a token counter. It verifies the fixed codec
and records prepared input only; provider delivery, source custody, semantic
coverage and complete-run savings require their existing independent owners.
"""
from __future__ import annotations

import re

from . import semantic_metadata_view as metadata


SCHEMA = "supervisor-representation-displacement@1"
MAX_RECEIPT_BYTES = 16_384
_MODES = {"common-bindings@1", "common-bindings-instruction@1"}
_ARMS = ("native", "candidate", "selected")
_COMPLETE_FIELDS = {arm + "_complete_" + name for arm in _ARMS
                    for name in ("sha256", "bytes", "proxy_tokens")}
_SELECTION_BASE_FIELDS = (_COMPLETE_FIELDS | metadata._FALSE_FIELDS | {
    "schema", "view_receipt_sha256", "selected_mode", "fallback_reason",
    "token_proxy", "candidate_only"})
_TOKEN_SELECTION_FIELDS = {
    "token_counter_status", "token_counter_identity", "token_counter_identity_sha256",
    "native_complete_tokens", "candidate_complete_tokens", "selected_complete_tokens",
    "selection_basis", "actual_provider_usage_verified", "provider_usage_reported",
    "cli_scaffolding_measured", "complete_run_token_savings_claimed"}
_FALSE_FIELDS = {
    "source_freshness_verified", "required_fact_coverage_verified", "semantic_alignment_verified",
    "ir_provenance_verified", "program_semantics_proved", "omission_authority",
    "proof_authority", "dispatch_authority", "execution_authority", "completion_authority",
    "publication_authority", "actual_provider_usage_verified", "provider_usage_reported",
    "provider_input_delivery_verified", "cli_scaffolding_measured",
    "complete_run_token_savings_claimed", "total_token_savings_qualified",
    "actual_avoided_reads_verified", "counterfactual_read_savings_claimed",
    "counter_measurement_replayed_here", "counter_identity_authenticated"}
_BINDING_DIGEST_FIELDS = {
    "task_id_sha256", "translation_cid_sha256", "scope_cid_sha256", "semantic_root_cid_sha256",
    "source_manifest_sha256", "native_core_sha256", "native_context_sha256",
    "native_suffix_sha256", "original_translated_semantic_sha256"}
_INSTRUCTION_FIELDS = {"instruction_receipt_sha256", "instruction_binding_sha256",
    "instruction_source_sha256", "instruction_source_bytes", "instruction_block_sha256",
    "instruction_block_bytes"}
_COUNTER_FIELDS = {"token_counter_status", "token_counter_identity_sha256",
    "token_counter_qualification", "token_counter_identity_field_sha256",
    "native_complete_tokens", "candidate_complete_tokens", "selected_complete_tokens"}
_FIELDS = (_COMPLETE_FIELDS | _FALSE_FIELDS | _BINDING_DIGEST_FIELDS | _INSTRUCTION_FIELDS
    | _COUNTER_FIELDS | {"schema", "observation_stage", "selection_schema", "requested_mode",
    "selected_mode", "fallback_reason", "treatment", "selection_basis", "token_proxy",
    "view_receipt_sha256", "selection_receipt_sha256", "provider_sha256", "model_sha256",
    "native_transport_span", "candidate_transport_span", "selected_transport_span",
    "unchanged_prefix_sha256", "unchanged_prefix_bytes", "unchanged_suffix_sha256",
    "unchanged_suffix_bytes", "candidate_complete_byte_reduction", "selected_complete_byte_reduction",
    "source_preservation", "required_source_bytes_preserved", "source_guidance",
    "proposed_avoided_reads"})


class RepresentationDisplacementError(ValueError):
    """A closed representation observation could not be verified."""


def _need(condition, reason):
    if not condition:
        raise RepresentationDisplacementError(reason)


def _digest(value):
    _need(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None,
          "invalid displacement digest")
    return value


def _integer(value, minimum=0, maximum=metadata.MAX_BYTES):
    _need(type(value) is int and minimum <= value <= maximum,
          "invalid displacement byte or token count")
    return value


def _text(value):
    _need(type(value) is str and value.strip() and metadata._byte_count(value) <= 4096,
          "invalid displacement provider or model identity")
    return value


def _copy(value, *, limit=metadata.MAX_BYTES):
    raw = metadata._json(value)
    _need(len(raw.encode("utf-8")) <= limit, "displacement receipt exceeds byte bound")
    return metadata._parse(raw)


def _complete(prompts):
    return {arm + "_complete_" + field: value for arm, prompt in prompts.items()
            for field, value in (("sha256", metadata._sha(prompt)),
                                 ("bytes", metadata._byte_count(prompt)),
                                 ("proxy_tokens", metadata._proxy(prompt)))}


def _recorded_counter(receipt, *, provider, model):
    """Join recorded @2 arithmetic without invoking or authenticating a counter."""
    status, identity = receipt["token_counter_status"], receipt["token_counter_identity"]
    _need(type(status) is str and status in {"counted", "failed", "invalid"},
          "invalid recorded displacement counter status")
    if status == "invalid":
        _need(identity is None and receipt["token_counter_identity_sha256"] is None,
              "invalid recorded counter carries an identity")
    else:
        fields = {"schema", "provider", "model", "tokenizer", "tokenizer_revision", "qualification",
                  "measurement_scope", "identity_caller_supplied", "provider_usage_verified"}
        _need(type(identity) is dict and set(identity) == fields
              and identity["schema"] == metadata.TOKEN_COUNTER_SCHEMA
              and identity["measurement_scope"] == metadata.TOKEN_MEASUREMENT_SCOPE
              and identity["identity_caller_supplied"] is True
              and identity["provider_usage_verified"] is False
              and type(identity["qualification"]) is str
              and identity["qualification"] in {"provider_tokenizer", "estimate"},
              "recorded displacement counter identity differs")
        for name in ("provider", "model", "tokenizer", "tokenizer_revision", "qualification"):
            _text(identity[name])
        _need(identity["provider"] == provider and identity["model"] == model,
              "recorded counter differs from provider or model")
        _need(_digest(receipt["token_counter_identity_sha256"]) == metadata._sha(metadata._json(identity)),
              "recorded counter identity digest differs")
    tokens = {arm + "_complete_tokens": receipt[arm + "_complete_tokens"] for arm in _ARMS}
    if status == "counted":
        for value in tokens.values():
            _integer(value, 1, 2**63 - 1)
    else:
        _need(all(value is None for value in tokens.values()),
              "unavailable counter carries token measurements")
    return {"token_counter_status": status,
        "token_counter_identity_sha256": receipt["token_counter_identity_sha256"],
        "token_counter_qualification": identity["qualification"] if identity else None,
        "token_counter_identity_field_sha256": ({name: metadata._sha(identity[name]) for name in
            ("provider", "model", "tokenizer", "tokenizer_revision")} if identity else None), **tokens}


def _selection(view, selection, prompts, *, requested, provider, model, instruction_block):
    receipt = selection.receipt
    base = _SELECTION_BASE_FIELDS | ({"requested_mode"} if requested != "common-bindings@1" else set())
    schema = receipt.get("schema")
    _need(type(schema) is str and schema in {metadata.SELECTION_SCHEMA, metadata.TOKEN_SELECTION_SCHEMA},
          "unsupported displacement selection schema")
    fields = base | (_TOKEN_SELECTION_FIELDS if schema == metadata.TOKEN_SELECTION_SCHEMA else set())
    _need(set(receipt) == fields, "displacement selection fields differ")
    if schema == metadata.SELECTION_SCHEMA:
        if requested == "common-bindings-instruction@1":
            from .semantic_instruction_view import select_instruction_view
            replay = select_instruction_view(view=view, native_complete_prompt=prompts["native"],
                candidate_complete_prompt=prompts["candidate"], instruction_block=instruction_block)
        else:
            replay = metadata.select_semantic_metadata_view(view=view,
                native_complete_prompt=prompts["native"], candidate_complete_prompt=prompts["candidate"])
        _need(metadata._json(receipt) == replay.receipt_json
              and selection.selected_prompt == replay.selected_prompt
              and type(selection.selected_mode) is str and selection.selected_mode == replay.selected_mode,
              "proxy displacement selection differs from exact replay")
        counter = {"token_counter_status": "not_supplied", "token_counter_identity_sha256": None,
            "token_counter_qualification": None, "token_counter_identity_field_sha256": None,
            **{arm + "_complete_tokens": None for arm in _ARMS}}
        return receipt, counter
    counter = _recorded_counter(receipt, provider=provider, model=model)
    status = counter["token_counter_status"]
    chosen = (status == "counted" and metadata._byte_count(prompts["candidate"]) < metadata._byte_count(prompts["native"])
              and counter["candidate_complete_tokens"] < counter["native_complete_tokens"])
    arm = "candidate" if chosen else "native"
    fallback = (None if chosen else "complete_input_not_smaller_under_bytes_and_token_counter"
                if status == "counted" else "complete_input_token_counter_" + status)
    expected = {"schema": schema, "view_receipt_sha256": metadata._sha(view.receipt_json),
        "selected_mode": requested if chosen else "legacy", "fallback_reason": fallback,
        "token_proxy": metadata.TOKEN_PROXY, "candidate_only": True, **_complete(prompts),
        **{name: False for name in metadata._FALSE_FIELDS},
        **{name: receipt[name] for name in ("token_counter_status", "token_counter_identity",
            "token_counter_identity_sha256", "native_complete_tokens", "candidate_complete_tokens")},
        "selected_complete_tokens": counter[arm + "_complete_tokens"], "selection_basis": "token_counter",
        "actual_provider_usage_verified": False, "provider_usage_reported": False,
        "cli_scaffolding_measured": False, "complete_run_token_savings_claimed": False}
    if requested != "common-bindings@1":
        expected["requested_mode"] = requested
    _need(metadata._json(receipt) == metadata._json(expected)
          and type(selection.selected_mode) is str and selection.selected_mode == expected["selected_mode"]
          and prompts["selected"] == prompts[arm], "recorded displacement selection differs")
    return receipt, counter


def observe_representation_displacement(*, view: metadata.SemanticMetadataView,
        selection: metadata.SemanticMetadataSelection, native_complete_prompt: str,
        candidate_complete_prompt: str, provider: str, model: str,
        instruction_block: str | None = None, instruction_receipt: dict | None = None,
        source_guidance: str | None = None) -> dict:
    """Observe existing prepared-input replacement; no counter is called here.

    @1 repeats pure proxy selection. @2 checks recorded identity, arithmetic and
    exact prepared strings without authenticating or replaying its measurement.
    The returned hashes omit source text, source paths and provider/model names.
    """
    try:
        _need(type(view) is metadata.SemanticMetadataView
              and type(selection) is metadata.SemanticMetadataSelection,
              "typed displacement view and selection required")
        _text(provider)
        _text(model)
        requested = ("common-bindings-instruction@1" if instruction_block is not None
                     or instruction_receipt is not None else "common-bindings@1")
        if requested == "common-bindings-instruction@1":
            from .semantic_instruction_view import project_instruction_view, restore_instruction_view
            rebuilt = project_instruction_view(view.native_prompt, instruction_block=instruction_block,
                                                instruction_receipt=instruction_receipt)
            restored = restore_instruction_view(provider_prompt=view.provider_prompt,
                receipt=view.receipt, instruction_block=instruction_block)
        else:
            rebuilt = metadata.project_semantic_metadata_view(view.native_prompt)
            restored = metadata.restore_semantic_metadata_view(provider_prompt=view.provider_prompt, receipt=view.receipt)
        _need(rebuilt == view and restored == view.native_prompt,
              "displacement differs from the fixed lossless projector")
        prompts = {"native": native_complete_prompt, "candidate": candidate_complete_prompt,
                   "selected": selection.selected_prompt}
        for prompt in prompts.values():
            _need(type(prompt) is str and 0 < metadata._byte_count(prompt) <= metadata.MAX_BYTES,
                  "displacement complete input exceeds text bound")
        _need(native_complete_prompt.count(view.native_prompt) == 1,
              "displacement native transport occurrence differs")
        prefix, suffix = native_complete_prompt.split(view.native_prompt)
        _need(candidate_complete_prompt == prefix + view.provider_prompt + suffix
              and candidate_complete_prompt.count(view.provider_prompt) == 1,
              "displacement changed literal complete-input wrappers")
        if instruction_block is not None:
            _need(suffix.count(instruction_block) == 1, "displacement verbatim instruction occurrence differs")
        receipt, counter = _selection(view, selection, prompts, requested=requested,
            provider=provider, model=model, instruction_block=instruction_block)
        native_wire, candidate_wire = metadata._parse(view.native_prompt), metadata._parse(view.provider_prompt)
        sources = native_wire["translated_semantic"].get("raw_sources")
        candidate_sources = candidate_wire["translated_semantic"].get("raw_sources")
        _need(type(sources) is dict and all(type(value) is str for value in sources.values()),
              "displacement native source bodies must remain inline")
        if requested == "common-bindings@1":
            _need(metadata._json(sources) == metadata._json(candidate_sources),
                  "displacement source bodies differ")
        instruction = {name: None for name in _INSTRUCTION_FIELDS}
        if instruction_block is not None:
            binding = view.receipt["instruction_binding"]
            instruction = {"instruction_receipt_sha256": metadata._sha(metadata._json(instruction_receipt)),
                "instruction_binding_sha256": metadata._sha(metadata._json(binding)),
                "instruction_source_sha256": binding["source_sha256"],
                "instruction_source_bytes": binding["source_bytes"],
                "instruction_block_sha256": binding["block_sha256"], "instruction_block_bytes": binding["block_bytes"]}
        guidance = None
        if source_guidance is not None:
            _need(type(source_guidance) is str and 0 < metadata._byte_count(source_guidance) <= metadata.MAX_BYTES
                  and suffix.count(source_guidance) == 1
                  and all(prompt.count(source_guidance) == 1 for prompt in prompts.values()),
                  "displacement retained source guidance occurrence differs")
            start = metadata._byte_count(suffix.split(source_guidance)[0])
            guidance = {"treatment": "additive_retained_guidance", "sha256": metadata._sha(source_guidance),
                "bytes": metadata._byte_count(source_guidance), "suffix_start_byte": start,
                "suffix_end_byte": start + metadata._byte_count(source_guidance)}
        selected = selection.selected_mode != "legacy"
        start = metadata._byte_count(prefix)
        spans = {arm + "_transport_span": {"start_byte": start,
            "end_byte": start + metadata._byte_count(text), "sha256": metadata._sha(text)}
            for arm, text in (("native", view.native_prompt), ("candidate", view.provider_prompt),
                             ("selected", view.provider_prompt if selected else view.native_prompt))}
        bindings = view.receipt
        output = {"schema": SCHEMA, "observation_stage": "router_prepared_input",
            "selection_schema": receipt["schema"], "requested_mode": requested,
            "selected_mode": selection.selected_mode, "fallback_reason": receipt["fallback_reason"],
            "treatment": "rendered_replacement" if selected else "native_fallback",
            "selection_basis": "bytes_and_proxy" if receipt["schema"] == metadata.SELECTION_SCHEMA else "recorded_token_counter",
            "token_proxy": metadata.TOKEN_PROXY, "view_receipt_sha256": metadata._sha(view.receipt_json),
            "selection_receipt_sha256": metadata._sha(selection.receipt_json),
            "provider_sha256": metadata._sha(provider), "model_sha256": metadata._sha(model),
            **{name + "_sha256": metadata._sha(bindings[name]) for name in
               ("task_id", "translation_cid", "scope_cid", "semantic_root_cid")},
            **{name: bindings[name] for name in _BINDING_DIGEST_FIELDS if name not in
               {"task_id_sha256", "translation_cid_sha256", "scope_cid_sha256", "semantic_root_cid_sha256"}},
            **_complete(prompts), **counter, **spans, **instruction,
            "unchanged_prefix_sha256": metadata._sha(prefix), "unchanged_prefix_bytes": start,
            "unchanged_suffix_sha256": metadata._sha(suffix), "unchanged_suffix_bytes": metadata._byte_count(suffix),
            "candidate_complete_byte_reduction": metadata._byte_count(prompts["native"]) - metadata._byte_count(prompts["candidate"]),
            "selected_complete_byte_reduction": metadata._byte_count(prompts["native"]) - metadata._byte_count(prompts["selected"]),
            "source_preservation": ("signed_instruction_duplicate_referenced_verbatim_block_retained"
                if selected and instruction_block is not None else "all_raw_sources_unchanged_inline"),
            "required_source_bytes_preserved": True, "source_guidance": guidance,
            "proposed_avoided_reads": [], **{name: False for name in _FALSE_FIELDS}}
        return validate_representation_displacement(output,
            selected_input_sha256=metadata._sha(prompts["selected"]),
            selected_input_bytes=metadata._byte_count(prompts["selected"]))
    except RepresentationDisplacementError:
        raise
    except (metadata.SemanticMetadataViewError, TypeError, ValueError, KeyError, UnicodeError, RecursionError) as error:
        raise RepresentationDisplacementError("displacement input verification failed") from error


def validate_representation_displacement(receipt, *, selected_input_sha256, selected_input_bytes) -> dict:
    """Check a closed historical observation and selected-input join.

    This checks shape and recorded arithmetic only. Independent audit must
    rebuild the fixed projector from retained inputs to verify the observation.
    """
    try:
        _need(type(receipt) is dict, "displacement receipt must be a dictionary")
        value = _copy(receipt, limit=MAX_RECEIPT_BYTES)
        _need(set(value) == _FIELDS and value["schema"] == SCHEMA
              and value["observation_stage"] == "router_prepared_input"
              and value["token_proxy"] == metadata.TOKEN_PROXY
              and value["required_source_bytes_preserved"] is True
              and type(value["proposed_avoided_reads"]) is list and not value["proposed_avoided_reads"]
              and all(value[name] is False for name in _FALSE_FIELDS),
              "displacement receipt schema or claims differ")
        requested, selected_mode = value["requested_mode"], value["selected_mode"]
        _need(type(requested) is str and requested in _MODES and type(selected_mode) is str
              and selected_mode in {requested, "legacy"}, "displacement representation mode differs")
        selected = selected_mode != "legacy"
        _need(value["treatment"] == ("rendered_replacement" if selected else "native_fallback"),
              "displacement treatment differs from selected mode")
        for name in _BINDING_DIGEST_FIELDS | {"view_receipt_sha256", "selection_receipt_sha256",
                "provider_sha256", "model_sha256", "unchanged_prefix_sha256", "unchanged_suffix_sha256"}:
            _digest(value[name])
        prefix = _integer(value["unchanged_prefix_bytes"])
        suffix = _integer(value["unchanged_suffix_bytes"])
        for arm in _ARMS:
            _digest(value[arm + "_complete_sha256"])
            size = _integer(value[arm + "_complete_bytes"], 1)
            _need(_integer(value[arm + "_complete_proxy_tokens"], 1) == (size + 3) // 4,
                  "displacement proxy arithmetic differs")
            span = value[arm + "_transport_span"]
            _need(type(span) is dict and set(span) == {"start_byte", "end_byte", "sha256"},
                  "displacement transport span fields differ")
            _digest(span["sha256"])
            _need(_integer(span["start_byte"]) == prefix
                  and prefix < _integer(span["end_byte"]) <= size
                  and span["end_byte"] + suffix == size,
                  "displacement transport span bounds differ")
        arm = "candidate" if selected else "native"
        _need(value["selected_transport_span"] == value[arm + "_transport_span"]
              and all(value["selected_complete_" + field] == value[arm + "_complete_" + field]
                      for field in ("sha256", "bytes", "proxy_tokens")),
              "displacement selected input arm differs")
        _need(_digest(selected_input_sha256) == value["selected_complete_sha256"]
              and _integer(selected_input_bytes, 1) == value["selected_complete_bytes"],
              "displacement differs from selected model input")
        for arm in ("candidate", "selected"):
            _need(_integer(value[arm + "_complete_byte_reduction"], -metadata.MAX_BYTES)
                  == value["native_complete_bytes"] - value[arm + "_complete_bytes"],
                  "displacement complete byte reduction differs")
        status = value["token_counter_status"]
        _need(type(status) is str and status in {"not_supplied", "counted", "failed", "invalid"},
              "displacement token counter status differs")
        token_fields = ("native_complete_tokens", "candidate_complete_tokens", "selected_complete_tokens")
        if status == "not_supplied":
            _need(value["selection_schema"] == metadata.SELECTION_SCHEMA
                  and value["selection_basis"] == "bytes_and_proxy", "proxy displacement selection basis differs")
            should_select = (value["candidate_complete_bytes"] < value["native_complete_bytes"]
                            and value["candidate_complete_proxy_tokens"] < value["native_complete_proxy_tokens"])
            fallback = None if should_select else "complete_input_not_smaller_under_bytes_and_proxy"
        else:
            _need(value["selection_schema"] == metadata.TOKEN_SELECTION_SCHEMA
                  and value["selection_basis"] == "recorded_token_counter", "recorded displacement selection basis differs")
            if status == "counted":
                for name in token_fields:
                    _integer(value[name], 1, 2**63 - 1)
                _need(value["selected_complete_tokens"] == value[arm + "_complete_tokens"],
                      "displacement selected token arm differs")
                should_select = (value["candidate_complete_bytes"] < value["native_complete_bytes"]
                                 and value["candidate_complete_tokens"] < value["native_complete_tokens"])
                fallback = None if should_select else "complete_input_not_smaller_under_bytes_and_token_counter"
            else:
                should_select, fallback = False, "complete_input_token_counter_" + status
        _need(selected is should_select and value["fallback_reason"] == fallback,
              "displacement fallback or selection arithmetic differs")
        if status != "counted":
            _need(all(value[name] is None for name in token_fields),
                  "unavailable displacement counter carries token counts")
        if status in {"not_supplied", "invalid"}:
            _need(all(value[name] is None for name in ("token_counter_identity_sha256",
                "token_counter_qualification", "token_counter_identity_field_sha256")),
                "unavailable displacement counter carries an identity")
        else:
            _digest(value["token_counter_identity_sha256"])
            _need(type(value["token_counter_qualification"]) is str
                  and value["token_counter_qualification"] in {"estimate", "provider_tokenizer"},
                  "displacement counter qualification differs")
            fields = value["token_counter_identity_field_sha256"]
            _need(type(fields) is dict and set(fields) == {"provider", "model", "tokenizer", "tokenizer_revision"},
                  "displacement counter digest fields differ")
            for digest in fields.values():
                _digest(digest)
            _need(fields["provider"] == value["provider_sha256"] and fields["model"] == value["model_sha256"],
                  "displacement counter provider or model digest differs")
        instruction = requested == "common-bindings-instruction@1"
        _need(value["source_preservation"] == ("signed_instruction_duplicate_referenced_verbatim_block_retained"
              if instruction and selected else "all_raw_sources_unchanged_inline"),
              "displacement source preservation profile differs")
        if instruction:
            for name in _INSTRUCTION_FIELDS:
                if name.endswith("_bytes"):
                    _integer(value[name], 1, 32_768 if name == "instruction_source_bytes" else metadata.MAX_BYTES)
                else:
                    _digest(value[name])
            _need(value["instruction_source_bytes"] < value["instruction_block_bytes"] <= suffix,
                  "displacement instruction source or block size differs")
        else:
            _need(all(value[name] is None for name in _INSTRUCTION_FIELDS),
                  "metadata displacement carries an instruction binding")
        guidance = value["source_guidance"]
        if guidance is not None:
            _need(type(guidance) is dict and set(guidance) == {"treatment", "sha256", "bytes",
                  "suffix_start_byte", "suffix_end_byte"}
                  and guidance["treatment"] == "additive_retained_guidance",
                  "displacement guidance fields differ")
            _digest(guidance["sha256"])
            size = _integer(guidance["bytes"], 1)
            start = _integer(guidance["suffix_start_byte"])
            _need(_integer(guidance["suffix_end_byte"]) == start + size <= suffix,
                  "displacement retained guidance span differs")
        return value
    except RepresentationDisplacementError:
        raise
    except (metadata.SemanticMetadataViewError, TypeError, ValueError, KeyError, UnicodeError, RecursionError) as error:
        raise RepresentationDisplacementError("displacement receipt verification failed") from error
