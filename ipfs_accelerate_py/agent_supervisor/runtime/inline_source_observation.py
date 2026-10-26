"""Observe literal whole-file slots in a caller-verified rendered user prompt.

The caller must already have verified the exact final input. This pure observer
does not authenticate that provenance, replay a producer, or establish provider
ingestion. Only explicitly supported transport schemas with literal
``raw_sources`` are observed; full transport validation remains the caller's
responsibility. Other source-like strings and undeclared references or suffix
blocks remain outside coverage. Output contains commitments and logical UTF-8 lengths, never source
text, source paths, invocation IDs or exception details.
"""
from __future__ import annotations

import hashlib
import json
import re

from . import rendered_input_census as census
from .inline_source_carriers import CARRIER_SCOPES, source_carrier
from .repository_source_read_observation import EVENT_SCHEMA, make_read_event, validate_read_event


SCHEMA = "inline-source-observation@1"
CARRIER_SCHEMA = "inline-source-observation@2"
WORLD_SCHEMA = "inline-source-observation@3"
WORLD_TRANSPORT = "repository-world-packet@1"
WORLD_SCOPE = "world_packet.selected_semantic_context.raw_sources"
MAX_SOURCES = 64
# The manifest can also describe deferred sources. Match the semantic codec's
# MAX_ITEMS without treating that broader population as inline content.
MAX_MANIFEST_ITEMS = 4096
MAX_INPUT_BYTES = 256_000
MAX_RESULT_BYTES = 32_768
SUPPORTED_TRANSPORTS = frozenset({
    "supervisor-semantic-router-input@1", "supervisor-semantic-router-input@2",
    "supervisor-semantic-metadata-view@1", "supervisor-semantic-working-set-view@1",
})
REASONS = frozenset({"observer_unavailable", "invalid_input", "advisory_mismatch", "invalid_json",
    "unsupported_transport", "unsupported_world_reference_source_carrier", "invalid_source_population",
    "invalid_world_source_carrier", "invalid_source_binding", "source_limit", "result_limit"})
_ENTRY_FIELDS = frozenset({"source_identity_sha256", "scope", "representation", "encoding",
                           "content_sha256", "content_bytes"})
_UNKNOWN = ("full_prompt_source_coverage", "provider_tokens", "provider_ingestion_verified", "wire_overlap_bytes")
_FALSE = ("raw_content_exported", "proof_authority", "completion_authority")
_FIELDS = frozenset({"schema", "status", "reason", "scope", "model_prompt_sha256", "model_prompt_bytes",
    "invocation_id_sha256", "transport_schema", "entries", "source_count", "logical_source_bytes",
    "logical_byte_scope", *_UNKNOWN, *_FALSE})


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and len(value) == 64 and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _enum(value, choices):
    return type(value) is str and value in choices


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _safe_utf8(value, limit):
    try:
        return census._utf8(value, limit)
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        return None


def unavailable_inline_sources(*, model_prompt, workspace_advisory, invocation_id, reason="observer_unavailable"):
    """Return an unavailable observation, retaining only independently valid pins.

    The advisory is deliberately not exported or inferred from other text. This
    constructor also supports the caller's bounded diagnostic-failure path.
    """
    model = _safe_utf8(model_prompt, MAX_INPUT_BYTES)
    identity = _safe_utf8(invocation_id, 512)
    return dict(schema=SCHEMA, status="unavailable",
        reason=reason if _enum(reason, REASONS) else "observer_unavailable",
        scope="translated_semantic.raw_sources", model_prompt_sha256=_sha(model) if model else None,
        model_prompt_bytes=len(model) if model else None,
        invocation_id_sha256=_sha(identity) if identity else None, transport_schema=None,
        entries=None, source_count=None, logical_source_bytes=None,
        logical_byte_scope="decoded_utf8_source_text",
        **dict.fromkeys(_UNKNOWN), **dict.fromkeys(_FALSE, False))


def _bounded_structure(value):
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > census.MAX_NODES or depth > census.MAX_DEPTH:
            raise ValueError("inline_source_structure_limit")
        if type(item) is dict:
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)


def observe_inline_sources(*, model_prompt, workspace_advisory, invocation_id):
    """Observe supported literal raw-source slots after the exact advisory.

    Known wrappers use their explicitly selected literal slots. The instruction
    wrapper additionally binds its exact relocated source block. Native/world
    inputs remain unavailable. An empty supported literal map is an observed
    zero for its slot only, not full-prompt coverage.
    """
    unavailable = unavailable_inline_sources(model_prompt=model_prompt, workspace_advisory=workspace_advisory,
        invocation_id=invocation_id, reason="invalid_input")
    advisory = _safe_utf8(workspace_advisory, 8192)
    if unavailable["model_prompt_sha256"] is None or unavailable["invocation_id_sha256"] is None or advisory is None:
        return unavailable
    if not model_prompt.startswith(workspace_advisory):
        return {**unavailable, "reason": "advisory_mismatch"}
    try:
        wire, _prefix, suffix = census._json_prefix(model_prompt[len(workspace_advisory):])
        _bounded_structure(wire)
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        return {**unavailable, "reason": "invalid_json"}
    transport = wire.get("schema")
    if _enum(transport, SUPPORTED_TRANSPORTS):
        semantic = wire.get("translated_semantic")
        if type(semantic) is not dict:
            return {**unavailable, "reason": "invalid_source_population"}
        sources, manifest = semantic.get("raw_sources"), semantic.get("manifest")
        schema, scope = SCHEMA, "translated_semantic.raw_sources"
    elif _enum(transport, CARRIER_SCOPES):
        try:
            sources, manifest, scope = source_carrier(wire, suffix)
        except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
            return {**unavailable, "reason": "invalid_source_population"}
        schema = CARRIER_SCHEMA
    else:
        return {**unavailable, "reason": "unsupported_transport"}
    return _observe_sources(unavailable, sources, manifest, schema=schema, scope=scope, transport=transport)


def observe_world_inline_sources(*, model_prompt, workspace_advisory, invocation_id):
    """Join only complete, explicitly typed semantic references in selected input.

    Full packet admission and delivery provenance remain the caller's work. This
    observer reads no repository state and never fetches omitted chunks.
    """
    unavailable = unavailable_inline_sources(model_prompt=model_prompt, workspace_advisory=workspace_advisory,
        invocation_id=invocation_id, reason="invalid_input")
    advisory = _safe_utf8(workspace_advisory, 8192)
    if unavailable["model_prompt_sha256"] is None or unavailable["invocation_id_sha256"] is None or advisory is None:
        return unavailable
    if not model_prompt.startswith(workspace_advisory):
        return {**unavailable, "reason": "advisory_mismatch"}
    try:
        from .repository_world_packet import GUIDANCE
        from .world_inline_source_observation import world_sources
        router = model_prompt[len(workspace_advisory):]
        preamble = GUIDANCE + "\n"
        if not router.startswith(preamble):
            raise ValueError("invalid_world_guidance")
        envelope, prefix, _suffix = census._json_prefix(router[len(preamble):])
        _bounded_structure(envelope)
        if prefix != json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False):
            raise ValueError("invalid_world_encoding")
        sources, manifest, scope = world_sources(envelope)
        if scope != WORLD_SCOPE:
            raise ValueError("invalid_world_scope")
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError) as error:
        reason = ("unsupported_world_reference_source_carrier"
                  if type(error) is ValueError and str(error) == "unsupported_world_reference_source_carrier"
                  else "invalid_world_source_carrier")
        return {**unavailable, "reason": reason}
    return _observe_sources(unavailable, sources, manifest, schema=WORLD_SCHEMA,
                            scope=WORLD_SCOPE, transport=WORLD_TRANSPORT)


def _observe_sources(unavailable, sources, manifest, *, schema, scope, transport):
    """One shared content/hash/identity boundary for every supported carrier."""
    if (type(sources) is not dict or type(manifest) is not dict
            or len(sources) > MAX_SOURCES or len(manifest) > MAX_MANIFEST_ITEMS
            or not set(sources) <= set(manifest)):
        return {**unavailable, "reason": "invalid_source_population"}
    entries = []
    total = 0
    for path, content in sources.items():
        binding = manifest[path]
        if type(binding) is not dict or not _digest(binding.get("sha256")):
            return {**unavailable, "reason": "invalid_source_binding"}
        raw = _safe_utf8(content, MAX_INPUT_BYTES)
        if raw is None or _sha(raw) != binding["sha256"]:
            return {**unavailable, "reason": "invalid_source_binding"}
        event = make_read_event(dict(generation=1, path=path), dict(generation=1, path=path,
            source_sha256=binding["sha256"], content=content, representation="original",
            encoding="utf-8", proof_authority=False), True)
        if event is None:
            return {**unavailable, "reason": "invalid_source_binding"}
        entries.append({key: event[key] for key in sorted(_ENTRY_FIELDS)})
        total += len(raw)
        if total > unavailable["model_prompt_bytes"]:
            return {**unavailable, "reason": "source_limit"}
    result = {**unavailable, "schema": schema, "scope": scope,
        "status": "observed", "reason": None, "transport_schema": transport,
        "entries": sorted(entries, key=lambda entry: entry["source_identity_sha256"]),
        "source_count": len(entries), "logical_source_bytes": total}
    try:
        return validate_inline_sources(result, model_prompt_sha256=result["model_prompt_sha256"],
            model_prompt_bytes=result["model_prompt_bytes"], invocation_id_sha256=result["invocation_id_sha256"])
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        return {**unavailable, "reason": "result_limit"}


def _pair(digest, size):
    return digest is None and size is None or _digest(digest) and _integer(size, 1, MAX_INPUT_BYTES)


def validate_inline_sources(value, *, model_prompt_sha256, model_prompt_bytes, invocation_id_sha256):
    """Validate reported manifest arithmetic and exact outer pins, then detach.

    Nullable pins are supported only for unavailable observations with malformed
    input. Validating shape and these commitments does not authenticate the
    caller's rendered-input verification or recover the private identity inputs.
    """
    if (not _pair(model_prompt_sha256, model_prompt_bytes)
            or invocation_id_sha256 is not None and not _digest(invocation_id_sha256)
            or type(value) is not dict or len(value) != len(_FIELDS) or set(value) != _FIELDS
            or not _enum(value["schema"], {SCHEMA, CARRIER_SCHEMA, WORLD_SCHEMA})
            or not _enum(value["status"], {"observed", "unavailable"})
            or value["logical_byte_scope"] != "decoded_utf8_source_text"
            or not _pair(value["model_prompt_sha256"], value["model_prompt_bytes"])
            or value["invocation_id_sha256"] is not None and not _digest(value["invocation_id_sha256"])
            or (value["model_prompt_sha256"], value["model_prompt_bytes"], value["invocation_id_sha256"])
                != (model_prompt_sha256, model_prompt_bytes, invocation_id_sha256)
            or any(value[key] is not None for key in _UNKNOWN)
            or any(value[key] is not False for key in _FALSE)):
        raise ValueError("invalid_inline_source_observation")
    if value["status"] == "unavailable":
        if (value["schema"] != SCHEMA or value["scope"] != "translated_semantic.raw_sources"
                or not _enum(value["reason"], REASONS) or value["transport_schema"] is not None
                or any(value[key] is not None for key in ("entries", "source_count", "logical_source_bytes"))):
            raise ValueError("invalid_inline_source_observation")
        return dict(value)
    if value["schema"] == SCHEMA:
        supported = (_enum(value["transport_schema"], SUPPORTED_TRANSPORTS)
                     and value["scope"] == "translated_semantic.raw_sources")
    elif value["schema"] == CARRIER_SCHEMA:
        supported = (_enum(value["transport_schema"], CARRIER_SCOPES)
                     and value["scope"] == CARRIER_SCOPES[value["transport_schema"]])
    else:
        supported = value["transport_schema"] == WORLD_TRANSPORT and value["scope"] == WORLD_SCOPE
    if (model_prompt_sha256 is None or invocation_id_sha256 is None or value["reason"] is not None
            or not supported
            or type(value["entries"]) is not list or len(value["entries"]) > MAX_SOURCES
            or not _integer(value["source_count"], 0, MAX_SOURCES) or value["source_count"] != len(value["entries"])
            or not _integer(value["logical_source_bytes"], 0, model_prompt_bytes)):
        raise ValueError("invalid_inline_source_observation")
    previous, total, sizes, entries = None, 0, {}, []
    for entry in value["entries"]:
        if (type(entry) is not dict or len(entry) != len(_ENTRY_FIELDS) or set(entry) != _ENTRY_FIELDS
                or entry["scope"] != "file" or entry["representation"] != "original" or entry["encoding"] != "utf-8"):
            raise ValueError("invalid_inline_source_entry")
        event = validate_read_event({"schema": EVENT_SCHEMA, "generation": 1, **entry})
        identity, digest, size = event["source_identity_sha256"], event["content_sha256"], event["content_bytes"]
        if (previous is not None and identity <= previous
                or digest in sizes and sizes[digest] != size):
            raise ValueError("invalid_inline_source_entry")
        previous, sizes[digest] = identity, size
        total += size
        entries.append(dict(entry))
    if total != value["logical_source_bytes"]:
        raise ValueError("invalid_inline_source_totals")
    if len(json.dumps(value, sort_keys=True, allow_nan=False).encode()) > MAX_RESULT_BYTES:
        raise ValueError("inline_source_observation_limit")
    return {**value, "entries": entries}
