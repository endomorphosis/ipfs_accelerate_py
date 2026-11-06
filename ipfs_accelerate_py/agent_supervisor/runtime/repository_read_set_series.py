"""Bound cumulative local rendered input across independent Grok inspections.

Actual calls start fresh native allocations; prior answers are never replayed
into later model inputs.
Opt-in reuse returns an earlier successful answer for exactly equal complete
input bytes inside this one series, without another allocation or native charge.
Local tokenizer/estimate allowances exclude native system/tool/history inputs
and outputs. Complete observed final envelopes can stop the next inspection;
they establish neither a native hard cap nor whole-task usage or billing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from . import context_delivery_accounting as accounting
from .repository_context_read_set import compile_repository_read_set
from .repository_read_set_provider import inspect_repository_read_set
from .repository_source_session import RepositorySourceSession
from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES, PromptTokenCounter

SCHEMA = "repository-read-set-series@1"
MAX_QUERIES = 8
MAX_COUNTER = (1 << 63) - 1
MAX_INSPECTION_BYTES = 1024 * 1024
MAX_SERIES_BYTES = 768 * 1024
MAX_COMPACT_INSPECTION_BYTES = 65536
_INSPECTION_FIELDS = ("schema", "invocation_id", "provider", "model", "status", "failure_reason",
    "model_prompt_sha256", "model_prompt_bytes", "response_sha256", "response_bytes", "backend_dispatches",
    "provider_callback_attempts", "backend_returned", "predispatch_read_set_rechecked", "closing_read_set_rechecked",
    "closing_requirements_checked", "usage", "native_rollout_usage", "native_provider_outcome",
    "usage_capture_status", "native_max_turns", "effective_toolset_verified", "containment_verified",
    "local_fallback_allowed", "cross_provider_fallback_allowed", "automatic_retry_allowed",
    "native_cumulative_cap_enforced", "billing_total_verified", "completion_authority")
_SOURCE_KEYS = ("schema", "session_id", "generation", "package_manifest_sha256",
    "previous_package_manifest_sha256", "source_tree_sha256", "datasets_head",
    "semantic_state_cid", "index_coverage", "representation", "proof_authority",
    "execution_authority", "completion_authority")


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha(text):
    return sha256(text.encode("utf-8")).hexdigest()


def _integer(value, maximum, reason):
    _need(type(value) is int and 1 <= value <= maximum, reason)


def _text(value, maximum, reason):
    _need(type(value) is str and value.strip() and len(value.encode("utf-8")) <= maximum, reason)


def _copy_inspection(value):
    _need(type(value) is dict, "series_inspection_mapping_required")
    # Traverse before serializing so malformed counters/graphs stay bounded.
    pending, nodes, characters = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(nodes <= 100000 and depth <= 32, "series_inspection_structure_bound")
        if type(item) is str:
            characters += len(item)
        elif type(item) is dict:
            _need(all(type(key) is str for key in item), "series_inspection_json_required")
            characters += sum(map(len, item))
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is int:
            _need(item.bit_length() <= 64, "series_inspection_integer_bound")
        elif type(item) is float:
            _need(math.isfinite(item), "series_inspection_json_required")
        else:
            _need(item is None or type(item) is bool, "series_inspection_json_required")
        _need(characters <= MAX_INSPECTION_BYTES, "series_inspection_byte_bound")
    raw = _json(value)
    _need(len(raw.encode()) <= MAX_INSPECTION_BYTES, "series_inspection_byte_bound")
    return json.loads(raw)


@dataclass(frozen=True)
class ReadSetInspectionSeries:
    selected_prompts: tuple[str, ...] = field(repr=False)
    response_texts: tuple[str, ...] = field(repr=False)
    receipt_json: str = field(repr=False)

    @property
    def receipt(self):
        return json.loads(self.receipt_json)


def inspect_repository_read_set_series(*, session, generation, queries, required_paths,
        required_texts, task_id, provider, model, workspace, token_counter,
        max_total_input_tokens, max_input_tokens, limit=8, timeout=1800, max_output_tokens=2048,
        max_observed_native_tokens_before_next_dispatch=None, max_prompt_bytes=MAX_MODEL_PROMPT_BYTES, requirements_guard=None,
        receipt_observer=None, reuse_identical_inputs=False) -> ReadSetInspectionSeries:
    """Use the genuine inspection consumer and debit every dispatched input once.

    An independent ordered attempt manifest includes refused/missing receipts.
    Unknown dispatched observations and any execution failure prevent later
    calls. Observers receive final metadata once; pending exceptions keep their
    exact identity even if accounting or the observer also fails.
    """
    _need(type(reuse_identical_inputs) is bool, "series_reuse_flag_invalid")
    _need(type(session) is RepositorySourceSession, "series_exact_source_session_required")
    _need(provider == "grok_cli" and model == "grok-4.7", "series_route_unsupported")
    _need(type(queries) in (list, tuple) and 1 <= len(queries) <= MAX_QUERIES, "series_query_population_bound")
    queries = tuple(queries)
    for query in queries:
        _text(query, 4096, "series_query_bound")
    _need(type(required_paths) in (list, tuple) and len(required_paths) <= 8, "series_required_paths_bound")
    _need(type(required_texts) in (list, tuple) and 1 <= len(required_texts) <= 16, "series_required_texts_bound")
    required_paths, required_texts = tuple(required_paths), tuple(required_texts)
    _need(all(type(path) is str for path in required_paths), "series_required_path_invalid")
    for text in required_texts:
        _text(text, 16384, "series_required_texts_bound")
    _need(sum(len(text.encode()) for text in required_texts) <= 16384, "series_required_texts_bound")
    _need(type(task_id) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@+-]{0,255}", task_id), "series_task_invalid")
    for value, maximum in ((generation, 8), (limit, 32), (timeout, 1800),
            (max_output_tokens, 16384), (max_prompt_bytes, MAX_MODEL_PROMPT_BYTES),
            (max_total_input_tokens, MAX_COUNTER), (max_input_tokens, MAX_COUNTER)):
        _integer(value, maximum, "series_budget_or_setting_invalid")
    if max_observed_native_tokens_before_next_dispatch is not None:
        _integer(max_observed_native_tokens_before_next_dispatch, MAX_COUNTER, "series_native_threshold_invalid")
    _need(isinstance(workspace, Path) and workspace == session.workspace
        and Path.cwd().resolve(strict=True) == workspace, "series_workspace_differs")
    _need(type(token_counter) is PromptTokenCounter
        and (token_counter.provider, token_counter.model) == (provider, model), "series_counter_binding_differs")
    for callback in (requirements_guard, receipt_observer):
        _need(callback is None or callable(callback), "series_callback_invalid")
    metadata = (token_counter.binding(), token_counter.count_kind, token_counter.compatibility_basis)
    _need(metadata[1] in ("tokenizer", "estimate"), "series_counter_metadata_invalid")
    for value in [*metadata[0].values(), metadata[2]]:
        _text(value, 4096, "series_counter_metadata_invalid")
        _need(value == value.strip() and all(char.isprintable() for char in value), "series_counter_metadata_invalid")
    estimate = metadata[1] == "estimate"
    baseline, counts = None, {}

    def counter_current():
        if requirements_guard is not None:
            requirements_guard()
        _need((token_counter.binding(), token_counter.count_kind, token_counter.compatibility_basis) == metadata,
            "series_counter_changed")
        _need(Path.cwd().resolve(strict=True) == workspace, "series_workspace_differs")

    def current():
        nonlocal baseline
        counter_current()
        status = session.status_summary()
        _need(status.get("source_current") is True and status.get("refresh_required") is False
            and status.get("generation") == generation, "series_source_not_current")
        bound = {key: status[key] for key in _SOURCE_KEYS}
        if baseline is None:
            baseline = json.loads(_json(bound))
        _need(bound == baseline, "series_source_binding_changed")
        _need(Path.cwd().resolve(strict=True) == workspace, "series_workspace_differs")

    def count(prompt):
        # The genuine compiler closes source/query fences after counting, and
        # N6 repeats them at the actual backend seam. Keep this inner check cheap.
        counter_current()
        value = token_counter.count(prompt)
        _integer(value, MAX_COUNTER, "series_counter_magnitude_bound")
        counter_current()
        digest = _sha(prompt)
        _need(digest not in counts or counts[digest] == value, "series_counter_not_repeatable")
        _need(digest in counts or len(counts) < 1024, "series_counter_population_bound")
        counts[digest] = value
        return value

    wrapped = PromptTokenCounter(**metadata[0], count_tokens=count,
        count_kind=metadata[1], compatibility_basis=metadata[2])
    attempts = [dict(ordinal=index + 1, query_sha256=_sha(query), query_bytes=len(query.encode()),
        status="not_attempted", reason=None, input_allowance=None, backend_dispatches=None,
        local_input_tokens=None, local_estimated_input_tokens=None, local_debit=None,
        local_debit_status="not_attempted", inspection=None, inspection_observation_status="not_observed",
        response_index=None, reuse_reference=None) for index, query in enumerate(queries)]
    receipt = dict(schema=SCHEMA, status="returned", stop_reason=None, task_id=task_id,
        provider=provider, model=model, workspace_binding_sha256=_sha(str(workspace)),
        source_binding=None, required_paths=list(required_paths),
        required_text_bindings=[dict(sha256=_sha(text), bytes=len(text.encode())) for text in required_texts],
        query_count=len(queries), max_callbacks=len(queries), attempts=attempts,
        max_total_input_tokens=max_total_input_tokens, max_input_tokens=max_input_tokens,
        max_observed_native_tokens_before_next_dispatch=max_observed_native_tokens_before_next_dispatch,
        token_count_kind=metadata[1], tokenizer_binding=metadata[0], tokenizer_compatibility_basis=metadata[2],
        cumulative_input_tokens=None if estimate else 0, cumulative_estimated_input_tokens=0 if estimate else None,
        remaining_local_input_allowance=max_total_input_tokens, local_accounting_complete=True,
        backend_dispatches=0, known_backend_dispatches=0, scoped_native_observation=None,
        scoped_native_total_tokens=None, native_usage_known_for_continuing=True,
        count_scope="complete_rendered_user_inputs_at_controlled_backend_callbacks_only",
        local_budget_basis="declared_local_estimate" if estimate else "declared_local_tokenizer",
        independent_attempt_manifest=True, fresh_independent_allocations=True, prior_response_replayed=False,
        reuse_identical_inputs=reuse_identical_inputs, reused_results=0,
        result_reuse_scope="this series only; exact complete input bytes and fixed series bindings",
        result_reuse_persistent=False, reused_results_are_new_provider_calls=False,
        fresh_allocations_scope="actual backend calls only; reused results create no allocation",
        first_native_dispatch_may_overshoot=True, hidden_native_requests_capped=False,
        native_cumulative_cap_enforced=False, whole_task_usage_complete=None, billing_total_verified=False,
        token_budget_guaranteed=False, task_acceptance_verified=False, total_session_token_savings=None,
        proof_authority=False, execution_authority=False, completion_authority=False)
    prompts, responses, native_rows, invocation_ids, cached_results = [], [], [], set(), []
    used, remaining, original_error, attempt, native_unknown = 0, max_total_input_tokens, None, None, False

    def native_summary():
        receipt["scoped_native_total_tokens"] = None
        receipt["native_usage_known_for_continuing"] = False
        try:
            summary = accounting.summarize_invocations(native_rows,
                unreceipted_provider_attempt=not receipt["local_accounting_complete"] or native_unknown)
            receipt["scoped_native_observation"] = summary
            complete = (not native_unknown and receipt["local_accounting_complete"] and len(native_rows) == receipt["known_backend_dispatches"]
                and summary["supplied_invocation_count"] == len(native_rows)
                and summary["complete_supplied_native_usage"] is True)
            receipt["native_usage_known_for_continuing"] = bool(complete)
            receipt["scoped_native_total_tokens"] = summary["complete_native_total_tokens"] if complete else None
        except Exception:
            receipt["scoped_native_observation"] = None
            receipt["native_usage_known_for_continuing"] = False

    def retain(attempt, observed, prepared):
        nonlocal used, remaining, native_unknown
        dispatched = 0
        try:
            _need(attempt["inspection_observation_status"] == "retained"
                and observed is not None and observed.get("schema") == "repository-read-set-inspection@1"
                and observed.get("provider") == provider and observed.get("model") == model,
                "series_inspection_binding_invalid")
            dispatched = observed.get("backend_dispatches")
            _need(type(dispatched) is int and dispatched in (0, 1), "series_dispatch_count_invalid")
            attempt["backend_dispatches"] = dispatched
            if dispatched:
                held = observed.get("read_set")
                _need(type(held) is dict and held == prepared.receipt
                    and observed.get("model_prompt_sha256") == held["model_prompt_sha256"]
                    and observed.get("model_prompt_bytes") == held["model_prompt_bytes"],
                    "series_dispatched_input_binding_invalid")
                amount = held["selected_estimated_input_tokens" if estimate else "selected_input_tokens"]
                _need(type(amount) is int and 0 < amount <= attempt["input_allowance"], "series_input_debit_invalid")
                used += amount
                remaining -= amount
                attempt.update(local_debit=amount, local_debit_status="charged")
                attempt["local_estimated_input_tokens" if estimate else "local_input_tokens"] = amount
                receipt["known_backend_dispatches"] += 1
            else:
                attempt.update(local_debit=0, local_debit_status="not_dispatched")
        except Exception:
            if attempt["local_debit_status"] != "charged":
                remaining -= attempt["input_allowance"]
                attempt["local_debit_status"] = "reservation_withheld"
            receipt["local_accounting_complete"] = False
            native_unknown = True
        else:
            if dispatched:
                try:
                    identity = observed.get("invocation_id")
                    _need(type(identity) is str and identity not in invocation_ids, "series_duplicate_invocation_identity")
                    invocation_ids.add(identity)
                    native_rows.append(accounting.observe_invocation(observed))
                except Exception:
                    native_unknown = True
        receipt["remaining_local_input_allowance"] = remaining
        receipt["backend_dispatches"] = receipt["known_backend_dispatches"] if receipt["local_accounting_complete"] else None
        field = "cumulative_estimated_input_tokens" if estimate else "cumulative_input_tokens"
        receipt[field] = used if receipt["local_accounting_complete"] else None
        native_summary()

    def reuse(prepared, attempt):
        if not reuse_identical_inputs or prepared is None:
            return False
        current()
        raw = prepared.selected_prompt.encode("utf-8")
        for cached in cached_results:
            # Digest equality cannot replace literal complete-input equality.
            if raw != cached["prompt_bytes"]:
                continue
            reference = cached["reference"]
            _need(_sha(cached["prompt"]) == reference["model_prompt_sha256"]
                and len(raw) == reference["model_prompt_bytes"]
                and _sha(cached["response"]) == reference["response_sha256"]
                and len(cached["response"].encode("utf-8")) == reference["response_bytes"],
                "series_cached_result_binding_differs")
            current()
            prompts.append(cached["prompt"])
            responses.append(cached["response"])
            attempt.update(status="reused", backend_dispatches=0, local_debit=0,
                local_debit_status="reused", response_index=len(responses),
                reuse_reference=dict(reference))
            receipt["reused_results"] += 1
            current()
            return True
        return False

    def prepare(options):
        try:
            return compile_repository_read_set(**options)
        except ValueError as error:
            if str(error) != "read_set_required_token_budget_exceeded":
                raise
            return None

    try:
        current()
        receipt["source_binding"] = baseline
        for query, attempt in zip(queries, attempts, strict=True):
            current()
            options = dict(session=session, generation=generation, query=query, required_paths=required_paths,
                required_texts=required_texts, task_id=task_id, provider=provider, model=model,
                workspace=workspace, token_counter=wrapped, max_input_tokens=max_input_tokens,
                limit=limit, max_prompt_bytes=max_prompt_bytes)
            prepared = prepare(options) if reuse_identical_inputs else None
            if reuse(prepared, attempt):
                continue
            reason = None
            if receipt["known_backend_dispatches"] and not receipt["native_usage_known_for_continuing"]:
                reason = "prior_native_usage_unknown"
            elif (receipt["scoped_native_total_tokens"] is not None
                    and max_observed_native_tokens_before_next_dispatch is not None
                    and receipt["scoped_native_total_tokens"] >= max_observed_native_tokens_before_next_dispatch):
                reason = "native_threshold_reached"
            elif remaining <= 0:
                reason = "local_allowance_exhausted"
            if reason is not None:
                attempt.update(status="refused", reason=reason, backend_dispatches=0,
                    local_debit=0, local_debit_status="not_dispatched")
                receipt.update(status="stopped", stop_reason=reason)
                break
            allowance = min(max_input_tokens, remaining)
            attempt["input_allowance"] = allowance
            options["max_input_tokens"] = allowance
            if not reuse_identical_inputs or allowance != max_input_tokens:
                prepared = prepare(options)
            if prepared is None:
                attempt.update(status="refused", reason="local_allowance_exhausted", backend_dispatches=0,
                    local_debit=0, local_debit_status="not_dispatched")
                receipt.update(status="stopped", stop_reason="local_allowance_exhausted")
                break
            if reuse(prepared, attempt):
                continue
            observed = None
            def observe(value):
                nonlocal observed
                try:
                    _need(observed is None, "series_multiple_inspection_receipts")
                    observed = _copy_inspection(value)
                    compact = {key: observed[key] for key in _INSPECTION_FIELDS if key in observed}
                    compact["original_inspection_receipt_sha256"] = _sha(_json(observed))
                    compact["original_read_set_receipt_sha256"] = _sha(_json(observed.get("read_set")))
                    compact["inspection_projection_only"] = True
                    compact["original_inspection_receipt_bytes"] = len(_json(observed).encode())
                    _need(len(_json(compact).encode()) <= MAX_COMPACT_INSPECTION_BYTES, "series_compact_inspection_bound")
                    attempt.update(inspection=compact, inspection_observation_status="retained")
                except Exception:
                    attempt["inspection_observation_status"] = "invalid"
            try:
                result = inspect_repository_read_set(**options, timeout=timeout,
                    max_output_tokens=max_output_tokens, requirements_guard=current, receipt_observer=observe)
            except BaseException:
                attempt["status"] = "failed"
                try:
                    retain(attempt, observed, prepared)
                except BaseException:
                    receipt["local_accounting_complete"] = False
                    receipt["native_usage_known_for_continuing"] = False
                    receipt.update(cumulative_input_tokens=None, cumulative_estimated_input_tokens=None,
                        remaining_local_input_allowance=remaining, backend_dispatches=None,
                        scoped_native_total_tokens=None)
                raise
            retain(attempt, observed, prepared)
            _need(attempt["inspection_observation_status"] == "retained" and observed == result.receipt
                and receipt["local_accounting_complete"], "series_inspection_receipt_unavailable")
            prompts.append(result.selected_prompt)
            responses.append(result.response_text)
            attempt.update(status="returned", response_index=len(responses))
            current()
            if reuse_identical_inputs:
                _need(type(result.response_text) is str and bool(result.response_text.strip())
                    and observed.get("status") == "returned"
                    and result.selected_prompt.encode("utf-8") == prepared.selected_prompt.encode("utf-8"),
                    "series_successful_result_binding_differs")
                cached_results.append(dict(prompt=result.selected_prompt,
                    prompt_bytes=result.selected_prompt.encode("utf-8"), response=result.response_text,
                    reference=dict(original_attempt_ordinal=attempt["ordinal"],
                        original_response_index=attempt["response_index"],
                        original_invocation_id=observed["invocation_id"],
                        model_prompt_sha256=observed["model_prompt_sha256"],
                        model_prompt_bytes=observed["model_prompt_bytes"],
                        response_sha256=observed["response_sha256"], response_bytes=observed["response_bytes"])))
        current()
    except BaseException as error:
        original_error = error
        receipt.update(status="failed", stop_reason="inspection_failed")
        receipt["remaining_local_input_allowance"] = remaining
        field = "cumulative_estimated_input_tokens" if estimate else "cumulative_input_tokens"
        receipt[field] = used if receipt["local_accounting_complete"] else None
        receipt["backend_dispatches"] = receipt["known_backend_dispatches"] if receipt["local_accounting_complete"] else None
        receipt["native_usage_known_for_continuing"] = False
        receipt["scoped_native_total_tokens"] = None
        if attempt is not None and attempt["status"] == "not_attempted":
            attempt["status"] = "failed"
        raise
    finally:
        try:
            encoded = _json(receipt)
            _need(len(encoded.encode()) <= MAX_SERIES_BYTES, "series_receipt_byte_bound")
            if receipt_observer is not None:
                receipt_observer(json.loads(encoded))
        except BaseException:
            if original_error is None:
                raise
    return ReadSetInspectionSeries(tuple(prompts), tuple(responses), encoded)
