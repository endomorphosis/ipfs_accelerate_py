"""Lossless provenance deferral over an already verified semantic transport.

Behavioral fields, unknown fields, raw source and native task context remain
literal. Only the closed provenance allowlist can move behind exact native
capsule references. This pure codec never reads source or contacts the broker;
the caller owns current-source and endpoint validation before dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Callable

from . import semantic_router_translation as native
from . import worker_source_client as source
from .worker_semantic_evidence_client import MAX_SELECTED_CAPSULES, MAX_SELECTED_CAPSULE_BYTES

PROFILE = "capsule-references@1"
VIEW_SCHEMA = "supervisor-semantic-working-set-view@1"
RECEIPT_SCHEMA = "supervisor-semantic-working-set-projection@1"
SELECTION_SCHEMA = "supervisor-semantic-working-set-selection@1"
REFERENCE_SCHEMA = "supervisor-semantic-capsule-reference@1"
DESCRIPTOR_SCHEMA = "supervisor-worker-source-descriptor@4"
EVIDENCE_MODE = "selected-capsules@1"
MAX_BYTES = native.MAX_BYTES
DEFERRED_FIELDS = frozenset({
    "schema", "semantic_index_schema", "extractor_version", "capsule_schema",
    "capsule_compiler_version", "version_cid", "source_cid", "symbol_fact_cid",
    "relevant_binding_projection_cid", "dependency_version_cids",
    "dependency_fact_cids", "dependency_link_ids",
})
INSTRUCTIONS = (
    "Capsules retain all behavioral and unknown fields. Only provenance fields "
    "are deferred to working_set.capsule_references. Bind the source descriptor "
    "session and verify capsule_sha256 on expansion. Native "
    "capsule identifiers are unaliased; this transport's translation_table still "
    "applies to retained $semantic_ref objects. Missing provenance is not an empty "
    "value, a proof, or permission to omit source. After source edits, baseline "
    "expansion refuses; request current candidate context instead."
)
_FLAGS = ("proof_authority", "execution_authority", "completion_authority",
    "omission_authority", "semantic_equivalence_claimed", "program_semantics_proved",
    "provider_token_usage_verified", "complete_run_token_savings_claimed")
_BINDING_FIELDS = frozenset({"descriptor_schema", "descriptor_sha256",
    "descriptor_session_id", "semantic_evidence_mode", "semantic_binding"})
_PROJECTION_FIELDS = frozenset({"schema", "requested_profile", "source_transport_schema", "binding",
    "translation_cid", "native_transport_sha256", "native_transport_bytes", "candidate_view_sha256",
    "candidate_view_bytes", "capsule_count", "capsule_native_bytes", "capsule_references_sha256", "deferred_field_count",
    "deferred_field_allowlist", "behavioral_fields_deferred", "raw_source_deferred",
    "provenance_retrieval_required", "endpoint_reachability_verified", "native_provider_input_tokens",
    "candidate_provider_input_tokens", "provider_token_usage", *_FLAGS})
_SELECTION_FIELDS = frozenset({"schema", "requested_profile", "selected_mode", "selected",
    "fallback_reason", "view_receipt", "selection_basis", "native_complete_sha256",
    "native_complete_bytes", "candidate_complete_sha256", "candidate_complete_bytes",
    "selected_complete_sha256", "selected_complete_bytes", "native_provider_input_tokens",
    "candidate_provider_input_tokens", "selected_provider_input_tokens", "provider_token_usage", *_FLAGS})


class SemanticWorkingSetError(ValueError):
    """A closed transport, provenance reference or exact inverse differs."""


def _need(condition, reason):
    if not condition:
        raise SemanticWorkingSetError(reason)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise SemanticWorkingSetError("invalid working-set JSON") from error


def _sha(value):
    return hashlib.sha256(value.encode("utf-8") if type(value) is str else value).hexdigest()


def _parse(text):
    _need(type(text) is str and len(text.encode("utf-8")) <= MAX_BYTES,
        "working-set text exceeds bound")
    try:
        value = native._parse(text)
        _need(_json(value) == text, "canonical working-set JSON required")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise SemanticWorkingSetError("invalid canonical working-set JSON") from error
    return value


def _encoded(encoded, *, current):
    _need(type(encoded) is native.EncodedSemanticPrompt
        and type(encoded.table) is native.SemanticTranslationTable
        and type(encoded.freshness_checked) is bool,
        "exact encoded semantic prompt required")
    if current:
        _need(encoded.freshness_checked is True, "current semantic encoder required")
    try:
        wire = _parse(encoded.provider_prompt)
        _need(wire.get("schema") == native.TRANSPORT_SCHEMA,
            "working set requires original semantic transport")
        _need(native._restore(encoded.provider_prompt, encoded.table) == encoded.native_prompt,
            "encoded semantic prompt does not round-trip")
        receipt = encoded.receipt
        _need(receipt.get("native_prompt_sha256") == _sha(encoded.native_prompt)
            and receipt.get("provider_prompt_sha256") == _sha(encoded.provider_prompt)
            and receipt.get("freshness_checked") is encoded.freshness_checked,
            "encoded semantic receipt differs")
        _wire, _suffix, _positions, _refs, text, _artifact = native._prompt_parts(encoded.native_prompt)
        semantic = _parse(text)
        capsules = semantic["capsules"]
        _need(type(capsules) is list and len(capsules) <= native.MAX_ITEMS
            and all(type(row) is dict for row in capsules), "bounded native capsules required")
        _need(len(wire["translated_semantic"]["capsules"]) == len(capsules),
            "translated capsule population differs")
    except (KeyError, AttributeError, TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise SemanticWorkingSetError("invalid encoded semantic prompt") from error
    return wire, capsules


def _semantic_binding(encoded, profile):
    _need(type(profile) is str and profile in source.PROFILES, "unsupported source profile")
    table = encoded.table.to_dict()
    return {**{key: table[key] for key in source.BINDING_FIELDS
        if key not in {"profile", "source_manifest_sha256"}}, "profile": profile,
        "source_manifest_sha256": _sha(_json(table["source_manifest"]))}


def _binding(encoded, binding):
    _need(type(binding) is dict and set(binding) == _BINDING_FIELDS,
        "closed working-set binding required")
    _need(binding["descriptor_schema"] == DESCRIPTOR_SCHEMA
        and binding["semantic_evidence_mode"] == EVIDENCE_MODE,
        "selected capsule descriptor required")
    for key, length in (("descriptor_sha256", 64), ("descriptor_session_id", 32)):
        value = binding[key]
        _need(type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value),
            "invalid working-set descriptor binding")
    semantic = binding["semantic_binding"]
    _need(type(semantic) is dict and set(semantic) == set(source.BINDING_FIELDS),
        "complete semantic descriptor binding required")
    _need(_json(semantic) == _json(_semantic_binding(encoded, semantic["profile"])),
        "source descriptor differs from semantic encoder")
    return json.loads(_json(binding))


@dataclass(frozen=True)
class SemanticWorkingSetView:
    native_prompt: str
    provider_prompt: str
    receipt_json: str
    dispatch_usable: bool
    historical_replay: bool
    _encoded: native.EncodedSemanticPrompt = field(repr=False, compare=False)
    _binding_json: str = field(repr=False, compare=False)

    @property
    def receipt(self):
        return _parse(self.receipt_json)


@dataclass(frozen=True)
class SemanticWorkingSetSelection:
    provider_prompt: str
    selected_mode: str
    receipt_json: str
    dispatch_usable: bool
    historical_replay: bool

    @property
    def receipt(self):
        return _parse(self.receipt_json)


def _project(encoded, binding, *, current):
    wire, capsules = _encoded(encoded, current=current)
    binding = _binding(encoded, binding)
    capsule_native_bytes = sum(len(_json(capsule).encode("utf-8")) for capsule in capsules)
    _need(len(capsules) <= MAX_SELECTED_CAPSULES
        and capsule_native_bytes <= MAX_SELECTED_CAPSULE_BYTES,
        "selected capsules exceed bounded expansion eligibility")
    references, deferred_count = [], 0
    for index, (capsule, translated) in enumerate(zip(capsules, wire["translated_semantic"]["capsules"])):
        # No defaults or empty-value elision: every removed value is recovered
        # from the complete, exact native capsule and its pinned byte digest.
        for key in DEFERRED_FIELDS:
            if key in translated:
                del translated[key]
                deferred_count += 1
        references.append(dict(schema=REFERENCE_SCHEMA, capsule_index=index,
            capsule_sha256=_sha(_json(capsule))))
    from .worker_semantic_evidence_client import guidance
    wire.update(schema=VIEW_SCHEMA, source_transport_schema=native.TRANSPORT_SCHEMA,
        working_set=dict(profile=PROFILE, descriptor_sha256=binding["descriptor_sha256"],
            descriptor_session_id=binding["descriptor_session_id"],
            capsule_references=references), working_set_instructions=INSTRUCTIONS + guidance())
    provider_prompt = _json(wire)
    _need(len(provider_prompt.encode("utf-8")) <= MAX_BYTES, "working-set view exceeds bound")
    receipt = dict(schema=RECEIPT_SCHEMA, requested_profile=PROFILE,
        source_transport_schema=native.TRANSPORT_SCHEMA, binding=binding,
        translation_cid=encoded.table.translation_cid,
        native_transport_sha256=_sha(encoded.provider_prompt),
        native_transport_bytes=len(encoded.provider_prompt.encode("utf-8")),
        candidate_view_sha256=_sha(provider_prompt), candidate_view_bytes=len(provider_prompt.encode("utf-8")),
        capsule_count=len(capsules), capsule_native_bytes=capsule_native_bytes,
        capsule_references_sha256=_sha(_json(references)),
        deferred_field_count=deferred_count, deferred_field_allowlist=sorted(DEFERRED_FIELDS),
        behavioral_fields_deferred=False, raw_source_deferred=False,
        provenance_retrieval_required=True, endpoint_reachability_verified=False,
        native_provider_input_tokens=None, candidate_provider_input_tokens=None,
        provider_token_usage=None, **{key: False for key in _FLAGS})
    return SemanticWorkingSetView(encoded.provider_prompt, provider_prompt, _json(receipt),
        current, not current, encoded, _json(binding))


def project_semantic_working_set(encoded, descriptor) -> SemanticWorkingSetView:
    """Bind an operational candidate to the caller-validated current endpoint.

    Descriptor validation here checks shape and semantic identity. The caller
    must independently load/authenticate the endpoint and recheck freshness.
    """
    _encoded(encoded, current=True)
    try:
        source.validate_descriptor(descriptor)
    except (ValueError, TypeError, KeyError) as error:
        raise SemanticWorkingSetError("invalid source descriptor") from error
    _need(descriptor.get("schema") == DESCRIPTOR_SCHEMA
        and descriptor.get("semantic_evidence_mode") == EVIDENCE_MODE,
        "selected capsule descriptor required")
    binding = dict(descriptor_schema=descriptor["schema"], descriptor_sha256=_sha(_json(descriptor)),
        descriptor_session_id=descriptor["session_id"], semantic_evidence_mode=descriptor["semantic_evidence_mode"],
        semantic_binding={key: descriptor[key] for key in source.BINDING_FIELDS})
    return _project(encoded, binding, current=True)


def replay_semantic_working_set_for_audit(encoded, receipt) -> SemanticWorkingSetView:
    """Reconstruct historical view bytes, never endpoint or dispatch authority."""
    _need(type(receipt) is dict and type(receipt.get("binding")) is dict,
        "retained working-set receipt required")
    view = _project(encoded, receipt["binding"], current=False)
    _need(_json(receipt) == view.receipt_json, "retained working-set receipt differs")
    return view


def restore_semantic_working_set(*, provider_prompt, receipt, encoded, descriptor,
        resolver: Callable[[dict], bytes]) -> str:
    """Resolve exact native capsules and recover the original encoded transport.

    The resolver returns a complete canonical capsule from the bound broker.
    It owns page/session validation. This inverse independently verifies byte
    identity, all retained fields, and every alias position before restoration.
    """
    expected = project_semantic_working_set(encoded, descriptor)
    _need(type(receipt) is dict and _json(receipt) == expected.receipt_json
        and provider_prompt == expected.provider_prompt, "working-set view or receipt differs")
    _need(callable(resolver), "exact capsule resolver required")
    wire = _parse(provider_prompt)
    original = _parse(encoded.provider_prompt)
    table = encoded.table.to_dict()
    aliases = {row["source_symbol_id"]: row["target_symbol_ids"][0] for row in table["entries"]}
    for reference in wire["working_set"]["capsule_references"]:
        raw = resolver(dict(reference))
        _need(type(raw) is bytes and 0 < len(raw) <= MAX_BYTES
            and _sha(raw) == reference["capsule_sha256"], "expanded native capsule digest differs")
        try:
            capsule = _parse(raw.decode("utf-8"))
        except UnicodeError as error:
            raise SemanticWorkingSetError("expanded native capsule is not UTF-8") from error
        _need(type(capsule) is dict, "expanded native capsule must be an object")
        index = reference["capsule_index"]
        for path in table["replacement_paths"]:
            if len(path) >= 3 and path[:2] == ["capsules", index]:
                value = native._get(capsule, path[2:])
                _need(type(value) is str and value in aliases, "expanded capsule alias differs")
                native._set(capsule, path[2:], {native.REF_KEY: aliases[value]})
        retained = {key: value for key, value in capsule.items() if key not in DEFERRED_FIELDS}
        _need(_json(retained) == _json(wire["translated_semantic"]["capsules"][index]),
            "expanded capsule retained values differ")
        _need(_json(capsule) == _json(original["translated_semantic"]["capsules"][index]),
            "expanded translated capsule differs")
        wire["translated_semantic"]["capsules"][index] = capsule
    for key in ("source_transport_schema", "working_set", "working_set_instructions"):
        del wire[key]
    wire["schema"] = native.TRANSPORT_SCHEMA
    restored = _json(wire)
    _need(restored == encoded.provider_prompt, "working-set inverse differs")
    return restored


def select_working_set(view, native_complete_prompt, candidate_complete_prompt) -> SemanticWorkingSetSelection:
    """Select only a strictly smaller complete input, including all wrappers.

    Byte reduction is not a tokenizer measurement or evidence of run savings.
    Historical views may be remeasured but never become dispatch-usable here.
    """
    _need(type(view) is SemanticWorkingSetView and type(view.dispatch_usable) is bool
        and view.historical_replay is (not view.dispatch_usable), "exact working-set view required")
    expected = _project(view._encoded, _parse(view._binding_json), current=view.dispatch_usable)
    _need((view.native_prompt, view.provider_prompt, view.receipt_json)
        == (expected.native_prompt, expected.provider_prompt, expected.receipt_json),
        "working-set candidate identity differs")
    for prompt in (native_complete_prompt, candidate_complete_prompt):
        _need(type(prompt) is str and len(prompt.encode("utf-8")) <= MAX_BYTES,
            "complete working-set input exceeds bound")
    _need(native_complete_prompt.count(view.native_prompt) == 1,
        "complete native transport occurrence differs")
    prefix, suffix = native_complete_prompt.split(view.native_prompt)
    _need(candidate_complete_prompt == prefix + view.provider_prompt + suffix
        and candidate_complete_prompt.count(view.provider_prompt) == 1,
        "complete working-set input changed a prefix or suffix")
    before, after = len(native_complete_prompt.encode("utf-8")), len(candidate_complete_prompt.encode("utf-8"))
    selected = after < before
    chosen = candidate_complete_prompt if selected else native_complete_prompt
    receipt = dict(schema=SELECTION_SCHEMA, requested_profile=PROFILE,
        selected_mode=PROFILE if selected else "legacy", selected=selected,
        fallback_reason=None if selected else "complete_input_not_strictly_smaller",
        view_receipt=view.receipt, selection_basis="complete_prompt_utf8_bytes",
        native_complete_sha256=_sha(native_complete_prompt), native_complete_bytes=before,
        candidate_complete_sha256=_sha(candidate_complete_prompt), candidate_complete_bytes=after,
        selected_complete_sha256=_sha(chosen), selected_complete_bytes=len(chosen.encode("utf-8")),
        native_provider_input_tokens=None, candidate_provider_input_tokens=None,
        selected_provider_input_tokens=None, provider_token_usage=None,
        **{key: False for key in _FLAGS})
    validate_working_set_selection_receipt(receipt)
    return SemanticWorkingSetSelection(chosen, receipt["selected_mode"], _json(receipt),
        view.dispatch_usable, view.historical_replay)


def validate_working_set_selection_receipt(receipt):
    """Validate closed public accounting without source, endpoint or model IO."""
    _need(type(receipt) is dict and set(receipt) == _SELECTION_FIELDS,
        "closed working-set selection receipt required")
    projected = receipt["view_receipt"]
    _need(type(projected) is dict and set(projected) == _PROJECTION_FIELDS,
        "closed working-set view receipt required")
    _need(receipt["schema"] == SELECTION_SCHEMA and projected["schema"] == RECEIPT_SCHEMA
        and receipt["requested_profile"] == projected["requested_profile"] == PROFILE
        and projected["source_transport_schema"] == native.TRANSPORT_SCHEMA
        and receipt["selection_basis"] == "complete_prompt_utf8_bytes",
        "working-set receipt profile differs")
    for obj in (receipt, projected):
        _need(all(obj[key] is False for key in _FLAGS), "working-set receipt grants authority")
        _need(all(obj[key] is None for key in ("native_provider_input_tokens",
            "candidate_provider_input_tokens", "provider_token_usage")),
            "working-set receipt invents token usage")
    _need(receipt["selected_provider_input_tokens"] is None,
        "working-set receipt invents selected token usage")
    _need(projected["behavioral_fields_deferred"] is False
        and projected["raw_source_deferred"] is False
        and projected["provenance_retrieval_required"] is True
        and projected["endpoint_reachability_verified"] is False
        and projected["deferred_field_allowlist"] == sorted(DEFERRED_FIELDS),
        "working-set receipt deferral policy differs")
    def integer(value, low, high):
        _need(type(value) is int and low <= value <= high, "invalid working-set byte or count value")
    def digest(value, length=64):
        _need(type(value) is str and re.fullmatch("[0-9a-f]{" + str(length) + "}", value),
            "invalid working-set digest")
    integer(projected["capsule_count"], 0, MAX_SELECTED_CAPSULES)
    integer(projected["capsule_native_bytes"], 0, MAX_SELECTED_CAPSULE_BYTES)
    _need((projected["capsule_count"] == 0) == (projected["capsule_native_bytes"] == 0)
        and projected["capsule_native_bytes"] >= 2 * projected["capsule_count"],
        "working-set capsule count and byte coverage differ")
    integer(projected["deferred_field_count"], 0, projected["capsule_count"] * len(DEFERRED_FIELDS))
    for obj, fields in ((receipt, ("native_complete", "candidate_complete", "selected_complete")),
                        (projected, ("native_transport", "candidate_view"))):
        for key in fields:
            integer(obj[key + "_bytes"], 1, MAX_BYTES)
            digest(obj[key + "_sha256"])
    digest(projected["capsule_references_sha256"])
    _need(type(projected["translation_cid"]) is str
        and 0 < len(projected["translation_cid"].encode("utf-8")) <= 512,
        "invalid working-set translation identity")
    binding = projected["binding"]
    _need(type(binding) is dict and set(binding) == _BINDING_FIELDS
        and binding["descriptor_schema"] == DESCRIPTOR_SCHEMA
        and binding["semantic_evidence_mode"] == EVIDENCE_MODE,
        "invalid working-set binding receipt")
    digest(binding["descriptor_sha256"])
    digest(binding["descriptor_session_id"], 32)
    semantic = binding["semantic_binding"]
    _need(type(semantic) is dict and set(semantic) == set(source.BINDING_FIELDS)
        and type(semantic["profile"]) is str and semantic["profile"] in source.PROFILES,
        "invalid working-set semantic binding")
    for key in ("semantic_sha256", "native_prompt_sha256", "source_manifest_sha256"):
        digest(semantic[key])
    for key in ("task_id", "scope_cid", "semantic_root_cid"):
        _need(type(semantic[key]) is str and 0 < len(semantic[key].encode("utf-8")) <= 512
            and not any(ord(char) < 32 for char in semantic[key]),
            "invalid working-set semantic identity")
    before, after = receipt["native_complete_bytes"], receipt["candidate_complete_bytes"]
    _need(before >= projected["native_transport_bytes"] and after >= projected["candidate_view_bytes"]
        and before - projected["native_transport_bytes"] == after - projected["candidate_view_bytes"],
        "working-set complete wrapper bytes differ")
    selected = after < before
    _need(receipt["selected"] is selected
        and receipt["selected_mode"] == (PROFILE if selected else "legacy")
        and receipt["fallback_reason"] == (None if selected else "complete_input_not_strictly_smaller"),
        "working-set selection differs from complete byte measurement")
    key = "candidate_complete" if selected else "native_complete"
    _need(receipt["selected_complete_bytes"] == receipt[key + "_bytes"]
        and receipt["selected_complete_sha256"] == receipt[key + "_sha256"],
        "working-set selected input binding differs")
    _need(len(_json(receipt).encode("utf-8")) <= MAX_BYTES, "working-set receipt exceeds bound")
    return json.loads(_json(receipt))
