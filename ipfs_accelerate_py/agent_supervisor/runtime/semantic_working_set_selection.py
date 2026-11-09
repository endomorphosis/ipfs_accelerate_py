"""Recommend one complete local input; never admit or dispatch a working set.

The existing @1 codec validates the semantic/deferred pair without changing its
byte policy. This @2 receipt compares every byte-eligible complete alternative
under one declared counter, with a separate canonical output-schema cost.
Raw/semantic task equivalence and current source custody remain caller-owned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re

from . import semantic_working_set as working
from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES, PromptTokenCounter

SCHEMA = "supervisor-semantic-working-set-selection@2"
POLICY = "complete-input-local-token-minimum@1"
REPRESENTATIONS = ("raw_native", "semantic_full", "working_set")
MAX_TOKEN_COUNT = (1 << 63) - 1
MAX_REPORT_BYTES = 65536
_ROW = frozenset({"model_prompt_sha256", "model_prompt_bytes", "byte_eligible",
    "exclusion_reason", "content_tokens", "input_tokens", "estimated_content_tokens",
    "estimated_input_tokens"})
_SCHEMA_ROW = frozenset({"canonical_json_sha256", "canonical_json_bytes",
    "content_tokens", "estimated_content_tokens"})
_BINDING = frozenset({"provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy"})
_FALSE = ("dispatch_usable", "source_freshness_verified", "endpoint_reachability_verified",
    "native_semantic_equivalence_verified", "provider_framing_measured",
    "provider_tokenizer_equivalence_verified", "provider_token_usage_verified",
    "token_budget_guaranteed", "execution_authority", "completion_authority")
_FIELDS = frozenset({"schema", "selection_policy", "provider", "model", "candidates",
    "view_validation_receipt", "historical_replay", "native_output_schema",
    "selected_representation", "selected_complete_sha256", "selected_complete_bytes",
    "selection_reason", "tie_break_order", "measurement_status", "measurement_failure_reason",
    "tokenizer_binding", "token_count_kind", "tokenizer_compatibility_basis",
    "counter_evaluation_count", "eligible_candidate_measurements_complete",
    "max_prompt_bytes", "max_input_tokens", "token_budget_basis", "recommendation_only",
    "working_set_endpoint_preflight_required", "provider_token_usage",
    "total_session_token_savings", *_FALSE})
_FAILURES = frozenset({"counter_failed", "counter_changed", "counter_not_repeatable",
    "counter_magnitude_exceeded"})


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ValueError("bounded canonical selection JSON required") from error


def _digest(raw):
    return sha256(raw).hexdigest()


def _integer(value, low=0, high=MAX_TOKEN_COUNT):
    _need(type(value) is int and low <= value <= high, "selection integer outside its bound")


def _identity(value, maximum=4096):
    _need(type(value) is str and value == value.strip()
        and re.fullmatch(r"[\x20-\x7e]{1," + str(maximum) + "}", value) is not None,
        "bounded explicit selection identity required")


def _hash(value):
    _need(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "selection SHA-256 required")


@dataclass(frozen=True)
class CompleteWorkingSetSelection:
    selected_prompt: str = field(repr=False)
    selected_representation: str
    receipt_json: str = field(repr=False)
    historical_replay: bool
    dispatch_usable: bool = False
    recommendation_only: bool = True

    @property
    def receipt(self):
        return json.loads(self.receipt_json)


def validate_complete_working_set_selection_receipt(receipt):
    """Check closed accounting/arithmetic, not source or tokenizer authenticity."""
    _need(type(receipt) is dict and set(receipt) == _FIELDS, "closed @2 selection receipt required")
    r = receipt
    _need(r["schema"] == SCHEMA and r["selection_policy"] == POLICY
        and r["tie_break_order"] == list(REPRESENTATIONS), "selection policy differs")
    for name in ("provider", "model"):
        _identity(r[name], 256)
    _integer(r["max_prompt_bytes"], 1, MAX_MODEL_PROMPT_BYTES)
    if r["max_input_tokens"] is not None:
        _integer(r["max_input_tokens"], 1)
    _need(type(r["historical_replay"]) is bool and r["recommendation_only"] is True
        and all(r[name] is False for name in _FALSE)
        and r["provider_token_usage"] is None and r["total_session_token_savings"] is None,
        "selection receipt grants unsupported authority")
    legacy = working.validate_working_set_selection_receipt(r["view_validation_receipt"])
    rows = r["candidates"]
    _need(type(rows) is dict and set(rows) == set(REPRESENTATIONS), "exact three selection candidates required")
    status = r["measurement_status"]
    _need(type(status) is str and status in {"measured", "estimated", "unavailable", "failed"},
        "selection measurement status differs")
    known = status in {"measured", "estimated"}
    kind, binding = r["token_count_kind"], r["tokenizer_binding"]
    if status == "unavailable":
        _need(binding is None and kind is None and r["tokenizer_compatibility_basis"] is None,
            "unavailable selection has a counter binding")
    else:
        _need(type(binding) is dict and set(binding) == _BINDING
            and type(kind) is str and kind in {"tokenizer", "estimate"}, "closed declared counter binding required")
        for value in binding.values():
            _identity(value)
        _identity(r["tokenizer_compatibility_basis"])
        _need((binding["provider"], binding["model"]) == (r["provider"], r["model"])
            and (not known or kind == ("estimate" if status == "estimated" else "tokenizer")),
            "selection counter route or quality differs")
    _need((status == "failed" and type(r["measurement_failure_reason"]) is str
            and r["measurement_failure_reason"] in _FAILURES)
        or (status != "failed" and r["measurement_failure_reason"] is None),
        "selection failure reason differs")
    _integer(r["counter_evaluation_count"], 0, 8)
    _need((status != "unavailable" or r["counter_evaluation_count"] == 0)
        and (not known or r["counter_evaluation_count"] >= 2), "selection evaluation count differs")
    schema = r["native_output_schema"]
    _need(type(schema) is dict and set(schema) == _SCHEMA_ROW, "closed native schema measurement required")
    _integer(schema["canonical_json_bytes"], 0, MAX_MODEL_PROMPT_BYTES)
    _need(schema["canonical_json_bytes"] == 0 or schema["canonical_json_bytes"] >= 2,
        "native schema object byte size differs")
    if schema["canonical_json_bytes"]:
        _hash(schema["canonical_json_sha256"])
    else:
        _need(schema["canonical_json_sha256"] is None, "absent schema has a digest")
    field = "estimated_content_tokens" if kind == "estimate" else "content_tokens"
    other = "content_tokens" if kind == "estimate" else "estimated_content_tokens"
    if known:
        _integer(schema[field], 1 if schema["canonical_json_bytes"] else 0)
        _need(schema[other] is None and (schema["canonical_json_bytes"] != 0 or schema[field] == 0),
            "schema exact and estimated counts differ")
    else:
        _need(schema["content_tokens"] is schema["estimated_content_tokens"] is None,
            "unavailable schema has partial counts")
    eligible = []
    for name in REPRESENTATIONS:
        row = rows[name]
        _need(type(row) is dict and set(row) == _ROW, "closed complete candidate row required")
        _hash(row["model_prompt_sha256"])
        _integer(row["model_prompt_bytes"], 1, working.MAX_BYTES)
        fits = row["model_prompt_bytes"] <= r["max_prompt_bytes"]
        _need(row["byte_eligible"] is fits
            and row["exclusion_reason"] == (None if fits else "complete_input_byte_ceiling"),
            "candidate byte eligibility differs")
        if fits:
            eligible.append(name)
        if known and fits:
            _integer(row[field], 1)
            total = row[field] + schema[field]
            _integer(total, 1)
            input_field = "estimated_input_tokens" if kind == "estimate" else "input_tokens"
            _need(row[input_field] == total and type(row[input_field]) is int
                and row[other] is None
                and row["input_tokens" if kind == "estimate" else "estimated_input_tokens"] is None,
                "candidate schema arithmetic or count quality differs")
        else:
            _need(all(row[key] is None for key in ("content_tokens", "input_tokens",
                "estimated_content_tokens", "estimated_input_tokens")), "candidate has partial or excluded counts")
    _need(bool(eligible), "no byte-eligible complete candidate")
    observed = {}
    for row in [rows[name] for name in eligible] + ([schema] if schema["canonical_json_bytes"] else []):
        digest = row.get("model_prompt_sha256", row.get("canonical_json_sha256"))
        size = row.get("model_prompt_bytes", row.get("canonical_json_bytes"))
        witness = (size, row[field] if known else None)
        _need(digest not in observed or observed[digest] == witness,
            "same complete content has conflicting observations")
        observed[digest] = witness
    if known:
        _need(r["counter_evaluation_count"] == len(eligible) + bool(schema["canonical_json_bytes"]) + len(observed),
            "complete input and schema repeatability coverage differs")
    for name, prefix in (("semantic_full", "native_complete"), ("working_set", "candidate_complete")):
        _need(rows[name]["model_prompt_sha256"] == legacy[prefix + "_sha256"]
            and rows[name]["model_prompt_bytes"] == legacy[prefix + "_bytes"],
            "selection differs from validated view and wrappers")
    _need(r["eligible_candidate_measurements_complete"] is known, "eligible measurement coverage differs")
    input_field = "estimated_input_tokens" if kind == "estimate" else "input_tokens"
    winner = min(eligible, key=lambda name: rows[name][input_field]) if known else eligible[0]
    reason = ("minimum_local_complete_input" if known else
        "priority_fallback_no_counter" if status == "unavailable" else "priority_fallback_counter_failed")
    _need(r["selected_representation"] == winner and r["selection_reason"] == reason
        and r["selected_complete_sha256"] == rows[winner]["model_prompt_sha256"]
        and type(r["selected_complete_bytes"]) is int
        and r["selected_complete_bytes"] == rows[winner]["model_prompt_bytes"]
        and r["working_set_endpoint_preflight_required"] is (winner == "working_set"),
        "selection winner or selected binding differs")
    budget = r["max_input_tokens"]
    basis = ("estimated_complete_input" if kind == "estimate" else "local_tokenizer_complete_input")
    _need(r["token_budget_basis"] == (basis if budget is not None else None)
        and (budget is None or known and rows[winner][input_field] <= budget),
        "selection budget is unmeasured or exceeded")
    encoded = _json(r).encode("utf-8")
    _need(len(encoded) <= MAX_REPORT_BYTES, "selection receipt exceeds its byte bound")
    return json.loads(encoded)


def select_complete_working_set(*, view, native_complete_prompt, semantic_complete_prompt,
        working_set_complete_prompt, provider, model, token_counter=None,
        native_output_schema=None, max_input_tokens=None,
        max_prompt_bytes=MAX_MODEL_PROMPT_BYTES):
    """Return a complete-input recommendation, with no dispatch admission.

    Count only byte-eligible complete inputs; add the separately counted schema
    once per alternative. Missing/failed counters keep the first byte-eligible
    priority alternative unless an explicit token budget requires refusal.
    """
    for value in (provider, model):
        _identity(value, 256)
    _integer(max_prompt_bytes, 1, MAX_MODEL_PROMPT_BYTES)
    if max_input_tokens is not None:
        _integer(max_input_tokens, 1)
    if token_counter is not None:
        _need(type(token_counter) is PromptTokenCounter
            and (token_counter.provider, token_counter.model) == (provider, model),
            "counter differs from pinned provider/model")
    texts = dict(zip(REPRESENTATIONS, (native_complete_prompt, semantic_complete_prompt,
        working_set_complete_prompt), strict=True))
    rows = {}
    for name, text in texts.items():
        _need(type(text) is str, "literal complete input required")
        try:
            raw = text.encode("utf-8")
        except UnicodeError as error:
            raise ValueError("literal complete UTF-8 input required") from error
        _need(1 <= len(raw) <= working.MAX_BYTES, "complete input exceeds its hard byte bound")
        fits = len(raw) <= max_prompt_bytes
        rows[name] = dict(model_prompt_sha256=_digest(raw), model_prompt_bytes=len(raw),
            byte_eligible=fits, exclusion_reason=None if fits else "complete_input_byte_ceiling",
            content_tokens=None, input_tokens=None, estimated_content_tokens=None, estimated_input_tokens=None)
    eligible = [name for name in REPRESENTATIONS if rows[name]["byte_eligible"]]
    _need(bool(eligible), "no byte-eligible complete candidate")
    legacy = working.select_working_set(view, semantic_complete_prompt, working_set_complete_prompt).receipt
    _need(native_output_schema is None or type(native_output_schema) is dict, "native schema object required")
    schema_text = None if native_output_schema is None else _json(native_output_schema)
    try:
        schema_raw = b"" if schema_text is None else schema_text.encode("utf-8")
    except UnicodeError as error:
        raise ValueError("native schema UTF-8 required") from error
    _need(len(schema_raw) <= MAX_MODEL_PROMPT_BYTES, "native schema exceeds its byte bound")
    metadata = None if token_counter is None else (
        token_counter.binding(), token_counter.count_kind, token_counter.compatibility_basis)
    if metadata is not None:
        for value in metadata[0].values():
            _identity(value)
        _need(type(metadata[1]) is str and metadata[1] in {"tokenizer", "estimate"},
            "declared counter quality required")
        _identity(metadata[2])
    status = "unavailable" if metadata is None else "estimated" if metadata[1] == "estimate" else "measured"
    schema = dict(canonical_json_sha256=_digest(schema_raw) if schema_raw else None,
        canonical_json_bytes=len(schema_raw), content_tokens=None, estimated_content_tokens=None)
    evaluations, failure, observations = 0, None, {}

    def count(text):
        nonlocal evaluations, failure
        if (token_counter.binding(), token_counter.count_kind, token_counter.compatibility_basis) != metadata:
            failure = "counter_changed"
            raise ValueError("counter observation refused")
        evaluations += 1
        amount = token_counter.count(text)
        if (token_counter.binding(), token_counter.count_kind, token_counter.compatibility_basis) != metadata:
            failure = "counter_changed"
            raise ValueError("counter observation refused")
        if amount > MAX_TOKEN_COUNT:
            failure = "counter_magnitude_exceeded"
            raise ValueError("counter observation refused")
        if text in observations and observations[text] != amount:
            failure = "counter_not_repeatable"
            raise ValueError("counter observation refused")
        observations[text] = amount
        return amount

    if metadata is not None:
        try:
            counts = {name: count(texts[name]) for name in eligible}
            schema_count = 0 if schema_text is None else count(schema_text)
            for text in tuple(observations):
                count(text)
            if any(amount + schema_count > MAX_TOKEN_COUNT for amount in counts.values()):
                failure = "counter_magnitude_exceeded"
                raise ValueError("counter observation refused")
        except Exception:
            status, failure = "failed", failure or "counter_failed"
        else:
            field = "estimated_content_tokens" if metadata[1] == "estimate" else "content_tokens"
            input_field = "estimated_input_tokens" if metadata[1] == "estimate" else "input_tokens"
            schema[field] = schema_count
            for name, amount in counts.items():
                rows[name][field], rows[name][input_field] = amount, amount + schema_count
    known = status in {"measured", "estimated"}
    _need(max_input_tokens is None or known, "token budget requires a working consistent pinned counter")
    field = "estimated_input_tokens" if metadata is not None and metadata[1] == "estimate" else "input_tokens"
    selected = min(eligible, key=lambda name: rows[name][field]) if known else eligible[0]
    _need(max_input_tokens is None or rows[selected][field] <= max_input_tokens,
        "selected complete input exceeds its token budget")
    receipt = dict(schema=SCHEMA, selection_policy=POLICY, provider=provider, model=model,
        candidates=rows, view_validation_receipt=legacy, historical_replay=view.historical_replay,
        native_output_schema=schema, selected_representation=selected,
        selected_complete_sha256=rows[selected]["model_prompt_sha256"],
        selected_complete_bytes=rows[selected]["model_prompt_bytes"],
        selection_reason="minimum_local_complete_input" if known else
            "priority_fallback_no_counter" if status == "unavailable" else "priority_fallback_counter_failed",
        tie_break_order=list(REPRESENTATIONS), measurement_status=status, measurement_failure_reason=failure,
        tokenizer_binding=None if metadata is None else metadata[0],
        token_count_kind=None if metadata is None else metadata[1],
        tokenizer_compatibility_basis=None if metadata is None else metadata[2],
        counter_evaluation_count=evaluations, eligible_candidate_measurements_complete=known,
        max_prompt_bytes=max_prompt_bytes, max_input_tokens=max_input_tokens,
        token_budget_basis=(("estimated_complete_input" if metadata[1] == "estimate" else
            "local_tokenizer_complete_input") if max_input_tokens is not None else None),
        recommendation_only=True, working_set_endpoint_preflight_required=selected == "working_set",
        provider_token_usage=None, total_session_token_savings=None, **{name: False for name in _FALSE})
    validate_complete_working_set_selection_receipt(receipt)
    return CompleteWorkingSetSelection(texts[selected], selected, _json(receipt), view.historical_replay)
