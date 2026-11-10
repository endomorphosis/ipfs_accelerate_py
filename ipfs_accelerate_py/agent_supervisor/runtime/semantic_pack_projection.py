"""Lossless pack defaults and duplicate-estimate recipes, without authority.

Only exact duplicated values are factored. References remain separate records;
roles, identifiers, source requirements, caveats and policies remain literal.
No estimator, packer, provider, source or storage operation is performed here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re

from ..context.context_contracts import CONTEXT_REFERENCE_SCHEMA, CONTEXT_CONTRACT_VERSION
from . import semantic_router_translation as semantic

MODE = "semantic-pack-view@1"
VIEW_SCHEMA = "supervisor-semantic-pack-view@1"
RECEIPT_SCHEMA = "supervisor-semantic-pack-projection@1"
SELECTION_SCHEMA = "supervisor-semantic-pack-selection@1"
ESTIMATE_RECIPE = "from-inner-pack@1"
MAX_BYTES = 2_000_000
MAX_DEPTH = 40
MAX_NODES = 100_000
MAX_ROWS = 4096
DEFAULT_FIELDS = {
    "schema": CONTEXT_REFERENCE_SCHEMA, "contract_version": CONTEXT_CONTRACT_VERSION,
    "repository_id": "", "tree_id": "", "path": "", "byte_count": 0,
}
INSTRUCTIONS = (
    "For translated_semantic.pack, apply semantic_pack_recipe before reading its references: "
    "each reference inherits all reference_defaults fields; its remaining fields stay literal. "
    "For token_estimate='from-inner-pack@1', reconstruct the omitted token_estimate as "
    "{totals:pack.token_totals,estimator_version:pack.estimator_version,total:sum(pack.token_totals.values())}, "
    "where pack is the inner pack object. No reference or requirement is removed. "
    "This lossless representation grants no proof, omission, freshness or execution authority."
)
_TRANSPORT_FIELDS = frozenset({"schema", "translation_cid", "translation_table", "native_context",
    "translated_semantic", "native_suffix", "instructions", "execution_authority", "completion_authority"})
_VIEW_FIELDS = _TRANSPORT_FIELDS | {"source_transport_schema", "semantic_pack_recipe", "semantic_pack_instructions"}
_FALSE_FIELDS = frozenset({"proof_authority", "execution_authority", "completion_authority",
    "omission_authority", "source_freshness_verified", "program_semantics_proved",
    "provider_token_usage_verified", "complete_run_token_savings_claimed", "raw_content_exported"})
_TRUE_FIELDS = frozenset({"representation_lossless", "reference_roles_preserved",
    "unknown_fields_preserved", "inner_pack_preserved", "identifier_representation_preserved"})
_COUNT_FIELDS = frozenset({"reference_count", "reference_default_fields", "reference_fields_factored"})
_DIGEST_FIELDS = frozenset({"native_prompt_sha256", "translation_table_sha256", "recipe_sha256"})
_BYTE_NAMES = ("native_transport", "candidate_view", "original_pack", "candidate_pack")
_RECEIPT_FIELDS = frozenset({"schema", "view_schema", "source_schema", "provider_token_usage",
    "token_estimate_factored", *_COUNT_FIELDS, *_DIGEST_FIELDS, *_FALSE_FIELDS, *_TRUE_FIELDS,
    *(name+suffix for name in _BYTE_NAMES for suffix in ("_sha256", "_bytes"))})
_SELECTION_FIELDS = frozenset({"schema", "requested_mode", "selected_mode", "fallback_reason",
    "selection_basis", "view_receipt", "view_receipt_sha256", "provider_token_usage", *_FALSE_FIELDS,
    *(name+suffix for name in ("native_complete", "candidate_complete", "selected_complete")
        for suffix in ("_sha256", "_bytes"))})


class SemanticPackProjectionError(ValueError):
    """Closed representation failure which never includes input contents."""


def _need(value, reason):
    if not value:
        raise SemanticPackProjectionError(reason)


def _bytes(text):
    try:
        _need(type(text) is str, "bounded pack text required")
        size = len(text.encode("utf-8"))
        _need(size <= MAX_BYTES, "pack text exceeds byte bound")
        return size
    except UnicodeError:
        raise SemanticPackProjectionError("bounded pack text required") from None


def _sha(text):
    _bytes(text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json(value):
    try:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        _bytes(text)
        return text
    except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError):
        raise SemanticPackProjectionError("invalid bounded pack JSON") from None


def _parse(text):
    _bytes(text)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _need(key not in result, "duplicate pack JSON key")
            result[key] = value
        return result
    def no_constant(_):
        raise SemanticPackProjectionError("nonfinite pack number")
    try:
        value = json.loads(text, object_pairs_hook=unique, parse_constant=no_constant)
        nodes = 0
        def bounded(item, depth):
            nonlocal nodes
            nodes += 1
            _need(depth <= MAX_DEPTH and nodes <= MAX_NODES, "pack JSON exceeds structure bound")
            _need(type(item) is not float or math.isfinite(item), "nonfinite pack number")
            if type(item) is dict:
                for child in item.values(): bounded(child, depth+1)
            elif type(item) is list:
                for child in item: bounded(child, depth+1)
        bounded(value, 0)
        _need(_json(value) == text, "canonical pack JSON required")
        return value
    except (ValueError, TypeError, RecursionError, UnicodeError, OverflowError):
        raise SemanticPackProjectionError("invalid bounded canonical pack JSON") from None


def _measure(name, text):
    return {name+"_sha256": _sha(text), name+"_bytes": _bytes(text)}


def _totals(value):
    _need(type(value) is dict and all(type(n) is int and n >= 0 for n in value.values()),
        "pack token totals require nonnegative integers")


def _estimate(value):
    _need(type(value) is dict, "pack estimate requires an object")
    if "totals" in value: _totals(value["totals"])
    if "estimator_version" in value:
        _need(type(value["estimator_version"]) is str and bool(value["estimator_version"]),
            "pack estimator version requires text")
    if "total" in value:
        _need(type(value["total"]) is int and value["total"] >= 0, "pack estimate total requires an integer")
    if "total" in value and "totals" in value:
        _need(value["total"] == sum(value["totals"].values()), "pack estimate total differs from categories")


def _pack(wire, *, projected=False):
    semantic_body = wire.get("translated_semantic")
    _need(type(semantic_body) is dict and type(semantic_body.get("pack")) is dict,
        "semantic pack object required")
    pack = semantic_body["pack"]
    _need(type(pack.get("pack")) is dict and type(pack.get("references")) is list
        and len(pack["references"]) <= MAX_ROWS, "pack contract and bounded references required")
    inner = pack["pack"]
    if "token_totals" in inner: _totals(inner["token_totals"])
    if "estimator_version" in inner:
        _need(type(inner["estimator_version"]) is str and bool(inner["estimator_version"]),
            "pack estimator version requires text")
    if "token_estimate" in pack: _estimate(pack["token_estimate"])
    else: _need(projected, "original pack token estimate required")
    for row in pack["references"]:
        _need(type(row) is dict, "pack references require objects")
        for key in ("schema", "repository_id", "tree_id", "path"):
            if key in row: _need(type(row[key]) is str, "pack reference envelope text differs")
        for key in ("contract_version", "byte_count"):
            if key in row:
                _need(type(row[key]) is int and row[key] >= (1 if key == "contract_version" else 0),
                    "pack reference envelope integer differs")
    return pack


def _transport(text):
    wire = _parse(text)
    _need(type(wire) is dict and set(wire) == _TRANSPORT_FIELDS
        and wire.get("schema") == semantic.TRANSPORT_SCHEMA
        and wire.get("execution_authority") is False and wire.get("completion_authority") is False,
        "original semantic transport required")
    _pack(wire)
    return wire


def _derived_estimate(inner):
    if "token_totals" not in inner or "estimator_version" not in inner:
        return None
    return dict(totals=inner["token_totals"], estimator_version=inner["estimator_version"],
        total=sum(inner["token_totals"].values()))


def _project(original_prompt):
    wire = _transport(original_prompt)
    pack = _pack(wire)
    original_pack = _json(pack)
    rows = pack["references"]
    defaults = {key: value for key, value in DEFAULT_FIELDS.items() if rows
        and all(key in row and type(row[key]) is type(value) and row[key] == value for row in rows)}
    for row in rows:
        for key in defaults: del row[key]
    derived = _derived_estimate(pack["pack"])
    factored = derived is not None and _json(pack["token_estimate"]) == _json(derived)
    if factored: del pack["token_estimate"]
    recipe = dict(reference_defaults=defaults, token_estimate=ESTIMATE_RECIPE if factored else None)
    candidate_pack = _json(pack)
    wire.update(schema=VIEW_SCHEMA, source_transport_schema=semantic.TRANSPORT_SCHEMA,
        semantic_pack_recipe=recipe, semantic_pack_instructions=INSTRUCTIONS)
    return _json(wire), dict(reference_count=len(rows), reference_default_fields=len(defaults),
        reference_fields_factored=len(rows)*len(defaults), token_estimate_factored=factored,
        recipe_sha256=_sha(_json(recipe)), **_measure("original_pack", original_pack),
        **_measure("candidate_pack", candidate_pack))


@dataclass(frozen=True)
class SemanticPackView:
    original_prompt: str
    provider_prompt: str
    receipt_json: str

    @property
    def receipt(self): return _parse(self.receipt_json)


@dataclass(frozen=True)
class SemanticPackSelection:
    selected_prompt: str
    selected_mode: str
    receipt_json: str

    @property
    def receipt(self): return _parse(self.receipt_json)


def project_semantic_pack(encoded: semantic.EncodedSemanticPrompt) -> SemanticPackView:
    """Project current typed input; this pure operation does not recheck sources."""
    _need(type(encoded) is semantic.EncodedSemanticPrompt and encoded.freshness_checked is True,
        "current typed semantic prompt required")
    try:
        _need(semantic._restore(encoded.provider_prompt, encoded.table) == encoded.native_prompt,
            "native semantic inverse differs")
        _transport(encoded.provider_prompt)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, UnicodeError, RecursionError):
        raise SemanticPackProjectionError("verified original semantic transport required") from None
    candidate, counts = _project(encoded.provider_prompt)
    receipt = dict(schema=RECEIPT_SCHEMA, view_schema=VIEW_SCHEMA, source_schema=semantic.TRANSPORT_SCHEMA,
        native_prompt_sha256=_sha(encoded.native_prompt), translation_table_sha256=_sha(encoded.table.payload_json),
        provider_token_usage=None, **counts, **{key: False for key in _FALSE_FIELDS},
        **{key: True for key in _TRUE_FIELDS}, **_measure("native_transport", encoded.provider_prompt),
        **_measure("candidate_view", candidate))
    result = SemanticPackView(encoded.provider_prompt, candidate, _json(receipt))
    _need(restore_semantic_pack(candidate, receipt) == encoded.provider_prompt,
        "semantic pack projection did not round-trip")
    return result


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _counts(receipt, names):
    _need(all(_digest(receipt[name+"_sha256"])
        and type(receipt[name+"_bytes"]) is int and 0 <= receipt[name+"_bytes"] <= MAX_BYTES
        for name in names), "pack digest or byte count differs")


def validate_projection_receipt(receipt):
    _need(type(receipt) is dict, "pack projection receipt required")
    value = _parse(_json(receipt))
    _need(set(value) == _RECEIPT_FIELDS and value["schema"] == RECEIPT_SCHEMA
        and value["view_schema"] == VIEW_SCHEMA and value["source_schema"] == semantic.TRANSPORT_SCHEMA
        and all(value[key] is False for key in _FALSE_FIELDS)
        and all(value[key] is True for key in _TRUE_FIELDS) and value["provider_token_usage"] is None,
        "pack receipt schema or authority differs")
    _counts(value, _BYTE_NAMES)
    _need(all(_digest(value[key]) for key in _DIGEST_FIELDS)
        and all(type(value[key]) is int and value[key] >= 0 for key in _COUNT_FIELDS)
        and type(value["token_estimate_factored"]) is bool
        and value["reference_count"] <= MAX_ROWS and value["reference_default_fields"] <= len(DEFAULT_FIELDS)
        and value["reference_fields_factored"] == value["reference_count"]*value["reference_default_fields"]
        and (value["reference_count"] > 0 or value["reference_default_fields"] == 0),
        "pack receipt count or binding differs")
    return value


def restore_semantic_pack(provider_prompt: str, receipt: dict) -> str:
    """Restore the exact original transport against an independently retained receipt."""
    retained = validate_projection_receipt(receipt)
    wire = _parse(provider_prompt)
    _need(type(wire) is dict and set(wire) == _VIEW_FIELDS and wire["schema"] == VIEW_SCHEMA
        and wire["source_transport_schema"] == semantic.TRANSPORT_SCHEMA
        and wire["semantic_pack_instructions"] == INSTRUCTIONS
        and all(retained[key] == value for key, value in _measure("candidate_view", provider_prompt).items()),
        "pack view differs from retained receipt")
    recipe = wire["semantic_pack_recipe"]
    _need(type(recipe) is dict and set(recipe) == {"reference_defaults", "token_estimate"}
        and type(recipe["reference_defaults"]) is dict
        and all(key in DEFAULT_FIELDS and type(value) is type(DEFAULT_FIELDS[key])
            and value == DEFAULT_FIELDS[key] for key, value in recipe["reference_defaults"].items())
        and (recipe["token_estimate"] is None or recipe["token_estimate"] == ESTIMATE_RECIPE),
        "pack reconstruction recipe differs")
    pack = _pack(wire, projected=True)
    for row in pack["references"]:
        _need(not set(row) & set(recipe["reference_defaults"]), "pack default would replace explicit reference data")
        row.update(recipe["reference_defaults"])
    if recipe["token_estimate"] == ESTIMATE_RECIPE:
        _need("token_estimate" not in pack, "pack estimate recipe would replace literal data")
        derived = _derived_estimate(pack["pack"])
        _need(derived is not None, "pack estimate recipe inputs missing")
        pack["token_estimate"] = derived
    else: _need("token_estimate" in pack, "literal pack estimate missing")
    wire["schema"] = wire.pop("source_transport_schema")
    del wire["semantic_pack_recipe"]
    del wire["semantic_pack_instructions"]
    original = _json(wire)
    _need(all(retained[key] == value for key, value in _measure("native_transport", original).items()),
        "pack native transport differs")
    reproduced, counts = _project(original)
    _need(reproduced == provider_prompt and all(retained[key] == value for key, value in counts.items()),
        "pack projection replay differs")
    return original


def validate_selection_receipt(receipt):
    _need(type(receipt) is dict, "pack selection receipt required")
    value = _parse(_json(receipt))
    _need(set(value) == _SELECTION_FIELDS and value["schema"] == SELECTION_SCHEMA
        and value["requested_mode"] == MODE and value["selection_basis"] == "complete_prompt_utf8_bytes"
        and all(value[key] is False for key in _FALSE_FIELDS) and value["provider_token_usage"] is None,
        "pack selection schema or authority differs")
    view = validate_projection_receipt(value["view_receipt"])
    _need(value["view_receipt_sha256"] == _sha(_json(view)), "pack selection view binding differs")
    _counts(value, ("native_complete", "candidate_complete", "selected_complete"))
    _need(value["native_complete_bytes"] >= view["native_transport_bytes"]
        and value["candidate_complete_bytes"] >= view["candidate_view_bytes"]
        and value["native_complete_bytes"]-view["native_transport_bytes"]
            == value["candidate_complete_bytes"]-view["candidate_view_bytes"],
        "pack complete wrapper byte counts differ")
    smaller = value["candidate_complete_bytes"] < value["native_complete_bytes"]
    _need(value["selected_mode"] == (MODE if smaller else "legacy")
        and value["fallback_reason"] == (None if smaller else "complete_input_not_smaller"),
        "pack selection differs from strict byte reduction")
    chosen = "candidate_complete" if smaller else "native_complete"
    _need(all(value["selected_complete"+suffix] == value[chosen+suffix]
        for suffix in ("_sha256", "_bytes")), "pack selected input differs")
    return value


def select_semantic_pack(view: SemanticPackView, native_complete_prompt: str,
        candidate_complete_prompt: str) -> SemanticPackSelection:
    """Select only smaller complete input with identical literal wrapper bytes."""
    _need(type(view) is SemanticPackView, "typed semantic pack view required")
    _need(restore_semantic_pack(view.provider_prompt, view.receipt) == view.original_prompt,
        "pack original input differs")
    _bytes(native_complete_prompt); _bytes(candidate_complete_prompt)
    _need(native_complete_prompt.count(view.original_prompt) == 1, "complete native transport occurrence differs")
    prefix, suffix = native_complete_prompt.split(view.original_prompt)
    _need(candidate_complete_prompt == prefix+view.provider_prompt+suffix
        and candidate_complete_prompt.count(view.provider_prompt) == 1, "pack complete input changed literal wrappers")
    smaller = _bytes(candidate_complete_prompt) < _bytes(native_complete_prompt)
    selected = candidate_complete_prompt if smaller else native_complete_prompt
    receipt = dict(schema=SELECTION_SCHEMA, requested_mode=MODE, selected_mode=MODE if smaller else "legacy",
        fallback_reason=None if smaller else "complete_input_not_smaller", selection_basis="complete_prompt_utf8_bytes",
        view_receipt=view.receipt, view_receipt_sha256=_sha(view.receipt_json), provider_token_usage=None,
        **{key: False for key in _FALSE_FIELDS}, **_measure("native_complete", native_complete_prompt),
        **_measure("candidate_complete", candidate_complete_prompt), **_measure("selected_complete", selected))
    validate_selection_receipt(receipt)
    return SemanticPackSelection(selected, receipt["selected_mode"], _json(receipt))


__all__ = ["SemanticPackProjectionError", "SemanticPackView", "SemanticPackSelection",
    "project_semantic_pack", "restore_semantic_pack", "select_semantic_pack",
    "validate_projection_receipt", "validate_selection_receipt"]
