"""Select a measured semantic representation without granting producer authority.

Callers own freshness, codec restoration and projection admission. This bounded
adapter compares their strings using the existing complete prompt selector; it
never dispatches a provider or adds local counts to native session usage.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from .router_implementation_runner import (
    MAX_MODEL_PROMPT_BYTES,
    PromptTokenCounter,
    render_model_prompt,
    select_model_prompt,
)

SCHEMA = "supervisor-semantic-candidate-selection@1"
LABELS = ("legacy", "v2", "projected_v2", "packed_v2", "packed_projected_v2")
MAX_CANDIDATES = 8


def select_semantic_candidate(*, native_prompt: str, candidates: list[tuple[str, str]],
                              suffix: str, purpose: str, workspace: Path,
                              provider: str, model: str,
                              token_counter: PromptTokenCounter | None = None,
                              max_input_tokens: int | None = None,
                              max_prompt_bytes: int = MAX_MODEL_PROMPT_BYTES) -> tuple:
    """Return complete selected wires, a receipt, and a candidate label or None.

    Only strictly smaller complete counts (or labelled estimates) select a semantic
    string. Equal counts keep legacy, V2 and role-view V2 ahead of their new
    packed alternatives. No existing preference is silently reordered.
    Any failed or inconsistent counter discards every count from this operation.
    Invalid/oversized candidate strings do not discard other valid candidates.
    Native remains the fallback, subject to the unchanged configured ceilings.
    No native/projected string identity establishes codec or source validity.
    """
    if type(candidates) is not list or len(candidates) > MAX_CANDIDATES:
        raise ValueError("bounded semantic candidate list required")
    if (type(max_prompt_bytes) is not int or not 1 <= max_prompt_bytes <= MAX_MODEL_PROMPT_BYTES
            or max_input_tokens is not None and (type(max_input_tokens) is not int or max_input_tokens < 1)):
        raise ValueError("unchanged exact prompt byte/token ceilings required")
    if type(native_prompt) is not str or type(suffix) is not str:
        raise ValueError("literal native prompt and suffix required")
    if len(native_prompt.encode()) > MAX_MODEL_PROMPT_BYTES or len(suffix.encode()) > MAX_MODEL_PROMPT_BYTES:
        raise ValueError("bounded native prompt and suffix required")
    if token_counter is not None and (
            not isinstance(token_counter, PromptTokenCounter)
            or token_counter.provider != provider or token_counter.model != model):
        raise ValueError("token counter differs from pinned provider/model")
    named = {}
    for row in candidates:
        if type(row) is not tuple or len(row) != 2:
            raise ValueError("closed label/prompt candidate pair required")
        label, prompt = row
        if type(label) is not str or label not in LABELS or label in named:
            raise ValueError("unique supported semantic candidate label required")
        named[label] = prompt
    native_wire = native_prompt + suffix
    native_model, native_advisory = render_model_prompt(
        prompt=native_wire, purpose=purpose, workspace=workspace)
    comparison, measured, baseline, failures, counter_errors = {}, {}, None, [], []
    count_field = token_counter.count_field if token_counter is not None else "input_tokens"
    counts_by_hash = {}
    pair_counts = {}

    def observe_count(text):
        # Observe the existing trusted counter through the selector's callback.
        # If selection refuses an oversized native fallback after measuring a
        # pair, its counts still participate in consistency/failure checks.
        try:
            count = token_counter.count(text)
        except Exception as error:
            failures.append("failed")
            counter_errors.append(type(error).__name__)
            raise
        digest = sha256(text.encode()).hexdigest()
        if digest in counts_by_hash and counts_by_hash[digest] != count:
            failures.append("inconsistent")
        counts_by_hash[digest] = count
        pair_counts[digest] = count
        return count

    observed_counter = (replace(token_counter, count_tokens=observe_count)
                        if token_counter is not None else None)
    for label in LABELS:
        if label not in named:
            continue
        prompt = named[label]
        try:
            prompt_size = len(prompt.encode()) if type(prompt) is str else None
        except UnicodeError:
            prompt_size = None
        if prompt_size is None or not prompt or prompt_size > MAX_MODEL_PROMPT_BYTES:
            comparison[label] = {"available": False, "unavailable_reason": "invalid_or_oversized_candidate",
                                 "model_prompt_sha256": None, "model_prompt_bytes": None, "input_tokens": None}
            if count_field != "input_tokens":
                comparison[label][count_field] = None
            continue
        candidate_model, _ = render_model_prompt(prompt=prompt + suffix, purpose=purpose,
                                                 workspace=workspace, semantic_transport=True)
        if len(candidate_model.encode()) > max_prompt_bytes:
            comparison[label] = {"available": False, "unavailable_reason": "complete_candidate_byte_ceiling",
                                 "model_prompt_sha256": None, "model_prompt_bytes": len(candidate_model.encode()),
                                 "input_tokens": None}
            if count_field != "input_tokens":
                comparison[label][count_field] = None
            continue
        # The existing selector measures both complete inputs. Apply the caller's
        # possibly smaller byte/token ceilings after the competition, so a native
        # fallback which exceeds that ceiling cannot mask a fitting alternative.
        pair_counts = {}
        try:
            selected = select_model_prompt(native_prompt=native_prompt, semantic_prompt=prompt,
                suffix=suffix, purpose=purpose, workspace=workspace, provider=provider, model=model,
                token_counter=observed_counter, max_prompt_bytes=MAX_MODEL_PROMPT_BYTES)
        except ValueError as error:
            if str(error) != "model prompt with all required instructions exceeds its byte bound":
                raise
            # This valid candidate fits, but this pair preferred a native wire
            # over the hard byte ceiling. Try every other variant first. Any
            # counter failure or drift observed here remains globally fatal.
            digest = sha256(candidate_model.encode()).hexdigest()
            comparison[label] = {"available": True, "unavailable_reason": None,
                                 "model_prompt_sha256": digest,
                                 "model_prompt_bytes": len(candidate_model.encode()),
                                 "input_tokens": None}
            comparison[label][count_field] = pair_counts.get(digest)
            continue
        receipt = selected[3]
        native = receipt["candidates"]["native"]
        candidate = receipt["candidates"]["semantic"]
        if baseline is None:
            baseline = deepcopy(receipt)
        comparison[label] = {"available": True, "unavailable_reason": None, **candidate}
        measured[label] = selected
        if receipt["measurement_status"] == "failed":
            failures.append("failed")
            counter_errors.append(receipt["measurement_error_type"])
        if receipt["measurement_status"] in {"measured", "estimated"}:
            for row in (native, candidate):
                digest, count = row["model_prompt_sha256"], row[count_field]
                if digest in counts_by_hash and counts_by_hash[digest] != count:
                    failures.append("inconsistent")
                counts_by_hash[digest] = count
    if baseline is None:
        baseline = deepcopy(select_model_prompt(native_prompt=native_prompt, semantic_prompt=None,
            suffix=suffix, purpose=purpose, workspace=workspace, provider=provider, model=model,
            token_counter=observed_counter, max_prompt_bytes=MAX_MODEL_PROMPT_BYTES)[3])
        if baseline["measurement_status"] == "failed":
            failures.append("failed")
            counter_errors.append(baseline["measurement_error_type"])
    if failures:
        for candidate in comparison.values():
            candidate["input_tokens"] = None
            if count_field != "input_tokens":
                candidate[count_field] = None
        for candidate in baseline["candidates"].values():
            candidate["input_tokens"] = None
            if count_field != "input_tokens":
                candidate[count_field] = None
        status = "failed"
        reason = "tokenizer_inconsistent" if "inconsistent" in failures else "tokenizer_failed"
        selected_label = None
    else:
        status = baseline["measurement_status"]
        native_count = baseline["candidates"]["native"].get(count_field)
        eligible = [label for label in LABELS if label in measured
                    and comparison[label].get(count_field) is not None
                    and comparison[label][count_field] < native_count
                    and (max_input_tokens is None or comparison[label][count_field] <= max_input_tokens)]
        selected_label = min(eligible, key=lambda label: comparison[label][count_field]) if eligible else None
        reason = (("semantic_strictly_fewer_estimated_content_tokens" if status == "estimated"
                   else "semantic_strictly_fewer_content_tokens") if selected_label is not None else
                  "tokenizer_unavailable" if status == "unavailable" and named else
                  "native_not_larger" if named else "semantic_not_requested")
    if selected_label is not None:
        selected = measured[selected_label]
        result = selected[:3]
        receipt = deepcopy(selected[3])
    else:
        result = (native_model, native_wire, native_advisory)
        receipt = baseline
        receipt["candidates"] = {"native": receipt["candidates"]["native"]}
    chosen = receipt["candidates"]["semantic" if selected_label is not None else "native"]
    if chosen["model_prompt_bytes"] > max_prompt_bytes:
        raise ValueError("complete selected prompt exceeds its byte bound")
    if max_input_tokens is not None:
        if status not in {"measured", "estimated"}:
            raise ValueError("complete prompt token budget requires a working consistent pinned counter")
        if chosen[count_field] > max_input_tokens:
            raise ValueError("complete selected prompt exceeds its token budget")
    receipt.update(
        candidate_selection_schema=SCHEMA,
        selected_representation="semantic" if selected_label is not None else "native",
        selected_candidate=selected_label, selection_reason=reason, measurement_status=status,
        measurement_error_type="InconsistentTokenCount" if "inconsistent" in failures else counter_errors[0] if counter_errors else None,
        selected_content_tokens=chosen["input_tokens"],
        selected_estimated_content_tokens=chosen.get("estimated_input_tokens"), comparison_candidates=comparison,
        max_input_tokens=max_input_tokens, max_prompt_bytes=max_prompt_bytes,
        token_budget_basis=("estimated_content_tokens" if count_field == "estimated_input_tokens"
                            else "local_tokenizer_content_tokens") if max_input_tokens is not None else None,
        token_budget_guaranteed=False,
        tie_break_order=list(LABELS), token_measurements_consistent=True if status in {"measured", "estimated"} else False if failures else None,
        requested_candidate_labels=list(named),
        codec_validity_verified=False, projection_coverage_verified=False,
        provider_called=False, total_session_token_savings=None,
        execution_authority=False, completion_authority=False,
    )
    return (*result, receipt, selected_label)
