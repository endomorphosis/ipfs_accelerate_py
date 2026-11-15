"""Compare two current full-task requests carrying identical residual evidence.

The full alternative retains the baseline's selected optional population. The
focused alternative is advisory retrieval, not a proved equivalent omission.
Local complete-prompt counts never establish native usage or session savings.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import re
import threading
import weakref

from ..context.context_compiler import ContextCompilationReceipt, ContextCompileResult
from ..context.context_contracts import ContextCapsule
from ..planning.residual_llm_packet import ResidualLlmPacket
from ..procedure_compiler.world_model import RepositoryWorldState
from ..proof.formal_verification_contracts import canonical_json_bytes
from .repository_residual_context import _wire, replay_repository_residual_context
from .repository_world_packet import (
    PreparedRepositoryWorldPacket, _plain, _residual_binding_arguments, _residual_recipe,
    prepare_repository_world_packet,
)
from .repository_world_packet_replay import _decode_canonical
from .repository_world_residual import PreparedRepositoryWorldResidualRequest, _current, _custody
from .supervisor_meta_index import public_replay_without_metadata

PAIR_SCHEMA = "repository-world-residual-alternatives@1"
SELECTION_SCHEMA = "supervisor-world-residual-selection@1"
MAX_COUNT = (1 << 63) - 1
_SEAL, _LOCK, _ISSUED = object(), threading.RLock(), weakref.WeakKeyDictionary()
_COUNTER_FIELDS = ("provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy")
_CORE = ("repository_id", "tree_id", "objective_id", "objective_revision", "policy_id",
         "policy_revision", "caller", "stage", "budget", "goal", "scope", "acceptance")
_COMPILER_FIELDS = ("budget", "estimator", "value_policy", "provider_max_input_bytes",
                    "budget_resolution", "effective_input_limit", "effective_budget")
_FALSE = ("provider_retained_parent", "optional_omission_equivalence_verified",
          "token_savings_established", "default_adoption_authorized", "proof_authority",
          "execution_authority", "completion_authority")
_PAIR_FIELDS = frozenset({"schema", "baseline_prompt_sha256", "baseline_prompt_bytes",
    "full_prompt_sha256", "full_prompt_bytes", "focused_prompt_sha256", "focused_prompt_bytes",
    "baseline_context_cid", "full_context_cid", "focused_context_cid", "residual_packet_id",
    "residual_context_id", "residual_context_sha256", "source_binding", "task_text_sha256",
    "required_population_sha256", "full_optional_population_sha256", "focused_optional_population_sha256",
    "required_reference_count", "full_optional_count", "focused_optional_count",
    "full_baseline_optional_preserved", "full_task_invariants_preserved", "identical_residual_context",
    "provider_calls", "provider_input_tokens", *_FALSE})
_SELECTION_FIELDS = frozenset({"schema", "alternatives_receipt_sha256", "provider", "model",
    "selected_representation", "selection_reason", "candidates", "measurement_status",
    "measurement_error_type", "tokenizer_binding", "count_scope", "native_output_schema_tokens",
    "native_output_schema_sha256", "selected_content_tokens", "selected_input_tokens", "max_input_tokens",
    "max_prompt_bytes", "provider_framing_measured", "provider_tokenizer_equivalence_verified",
    "total_session_token_savings", "semantic_equivalence_verified", "omission_authority", "completion_authority"})


class RepositoryWorldResidualSelectionError(ValueError):
    """Closed diagnostics without prompt, source, counter or exception bodies."""


def _need(condition, code):
    if not condition:
        raise RepositoryWorldResidualSelectionError(code)


def _bytes(value):
    return _wire(_plain(value))


def _contract_bytes(value):
    # Native contracts can retain str-Enum objects in to_record(). Use their
    # established JSON spelling internally; external literal guards stay strict.
    raw = canonical_json_bytes(value)
    _need(len(raw) <= 2 * 1024 * 1024, "residual_native_contract_bound")
    return raw


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _hash(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _text(value):
    return type(value) is str and bool(value) and value.strip() == value and len(value) <= 1024 and "\x00" not in value


def _integer(value, minimum=0, maximum=MAX_COUNT):
    return type(value) is int and minimum <= value <= maximum


def _artifact(packet):
    _custody(packet)
    raw = packet._replay_material_json.encode("utf-8", "strict")
    return _decode_canonical(raw, _sha(raw))


def _required(capsule):
    return {ref.reference_id: ref.canonical_bytes() for ref in capsule.evidence if ref.required}


def _population(capsule, required):
    return [dict(reference_id_sha256=_sha(ref.reference_id.encode()), reference_content_id=ref.content_id)
            for ref in sorted(capsule.evidence, key=lambda ref: ref.reference_id) if ref.required is required]


def _arguments(recipe):
    result = dict(recipe)
    result["current_world"] = (None if recipe["current_world"] is None
        else RepositoryWorldState.from_dict(recipe["current_world"]))
    for name in ("unresolved_obligations", "validation_commands", "proof_obligations",
                 "acceptance_ids", "allowed_effects", "write_paths"):
        result[name] = tuple(result[name])
    return result


class PreparedRepositoryWorldResidualAlternatives:
    """Exact-issued frozen pair; neither getter grants freshness or dispatch."""

    __slots__ = ("_seal", "_baseline", "_full", "_focused", "_focus_request", "_focus_refs",
                 "_compiler", "_compiler_refs", "_estimator_refs", "_capsule_pins", "_focus_compilation_pin",
                 "_receipt_bytes", "__weakref__")

    def __init__(self, seal, *, baseline, full, focused, focus_request, receipt_bytes):
        _need(seal is _SEAL and type(self) is PreparedRepositoryWorldResidualAlternatives,
              "residual_alternatives_original_issuance_required")
        values = dict(_seal=seal, _baseline=baseline, _full=full, _focused=focused,
            _focus_request=focus_request,
            _focus_refs=tuple(getattr(focus_request, field.name) for field in fields(PreparedRepositoryWorldResidualRequest)),
            _compiler=baseline._adapter.compiler, _receipt_bytes=receipt_bytes)
        values["_compiler_refs"] = tuple((name, getattr(values["_compiler"], name)) for name in _COMPILER_FIELDS)
        estimator = values["_compiler"].estimator
        values["_estimator_refs"] = tuple((name, getattr(estimator, name)) for name in ("_tokenizer", "_chars_per_token"))
        values["_capsule_pins"] = tuple(_sha(ContextCapsule.canonical_bytes(packet._capsule)) for packet in (baseline, full, focused))
        values["_focus_compilation_pin"] = _sha(_contract_bytes(dict(
            receipt=focus_request.compilation.receipt.to_record(),
            decisions=[decision.to_dict() for decision in focus_request.compilation.decisions])))
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, _name, _value):
        raise AttributeError("immutable residual alternatives")

    def __delattr__(self, _name):
        raise AttributeError("immutable residual alternatives")

    def _seal_material(self):
        return tuple(getattr(self, name) for name in self.__slots__ if name != "__weakref__")

    def _retained(self):
        _need(type(self) is PreparedRepositoryWorldResidualAlternatives and getattr(self, "_seal", None) is _SEAL,
              "residual_alternatives_original_required")
        with _LOCK:
            original = _ISSUED.get(self)
        current = PreparedRepositoryWorldResidualAlternatives._seal_material(self)
        _need(original is not None and len(original) == len(current)
              and all(first is second for first, second in zip(original, current)),
              "residual_alternatives_not_issued_or_changed")
        _need(not set(self._focus_request.__dict__) & {"receipt"}, "residual_focus_method_shadowed")
        _need(all(getattr(self._focus_request, field.name) is retained for field, retained in
                  zip(fields(PreparedRepositoryWorldResidualRequest), self._focus_refs)),
              "residual_focus_request_changed")
        for packet in (self._baseline, self._full, self._focused):
            _custody(packet)
            _need(packet._adapter is self._baseline._adapter
                  and packet._adapter.compiler is self._compiler, "residual_alternatives_owner_changed")
        _need(all(getattr(self._compiler, name) is retained for name, retained in self._compiler_refs),
              "residual_alternatives_compiler_changed")
        _need(all(getattr(self._compiler.estimator, name) is retained for name, retained in self._estimator_refs),
              "residual_alternatives_estimator_changed")
        _need(tuple(_sha(ContextCapsule.canonical_bytes(packet._capsule)) for packet in
                    (self._baseline, self._full, self._focused)) == self._capsule_pins,
              "residual_alternatives_capsule_changed")
        _need(_sha(_contract_bytes(dict(receipt=self._focus_request.compilation.receipt.to_record(),
            decisions=[decision.to_dict() for decision in self._focus_request.compilation.decisions])))
            == self._focus_compilation_pin, "residual_alternatives_focus_compilation_changed")

    @property
    def full_packet(self):
        PreparedRepositoryWorldResidualAlternatives._retained(self)
        return self._full

    @property
    def focused_packet(self):
        PreparedRepositoryWorldResidualAlternatives._retained(self)
        return self._focused

    def receipt(self):
        PreparedRepositoryWorldResidualAlternatives._retained(self)
        return json.loads(self._receipt_bytes)


def prepare_repository_world_residual_alternatives(*, baseline, focused):
    """Retain the original selected population with the same residual failure.

    The original compilation receipt is reverified by its live compiler. The
    fallback optional population is not recompiled and no parent slot is added.
    Native currentness checks still repeat their recorded source queries. No
    old failure-free input can become the fallback; oversized fallback refuses.
    """
    _need(type(baseline) is PreparedRepositoryWorldPacket
          and type(focused) is PreparedRepositoryWorldResidualRequest,
          "residual_alternatives_exact_inputs_required")
    _need(not set(focused.__dict__) & {"receipt"}, "residual_focus_method_shadowed")
    candidate = focused.packet
    with public_replay_without_metadata():
        before, after = _artifact(baseline), _artifact(candidate)
        recipe, focused_recipe = before["recipe"], after["recipe"]
        _need("residual_context" not in recipe and "residual_context" in focused_recipe,
              "residual_alternatives_lineage_required")
        _need(_bytes(recipe) == _bytes({key: value for key, value in focused_recipe.items()
                                      if key != "residual_context"}), "residual_alternatives_recipe_changed")
        _need(candidate._adapter is baseline._adapter and candidate._workspace == baseline._workspace
              and candidate._generation == baseline._generation
              and candidate._task_id == baseline._task_id
              and candidate._intent_world_observation is baseline._intent_world_observation,
              "residual_alternatives_owner_mismatch")
        candidate_core, baseline_core = candidate._capsule.to_record(), baseline._capsule.to_record()
        _need(all(_contract_bytes(candidate_core[name]) == _contract_bytes(baseline_core[name])
                  for name in _CORE)
              and _contract_bytes(candidate._capsule.authority["task"]) == _contract_bytes(baseline._capsule.authority["task"])
              and _required(candidate._capsule) == _required(baseline._capsule),
              "residual_alternatives_task_or_required_changed")
        _need(type(focused.compilation) is ContextCompileResult
              and focused.compilation.capsule is candidate._capsule
              and _contract_bytes(focused.compilation.receipt.to_record()) == _bytes(after["compilation_receipt"]),
              "residual_alternatives_focus_compilation_changed")
        task_text = recipe["task_text"]
        baseline_receipt = _current(baseline, task_text=task_text)
        candidate_receipt = _current(candidate, task_text=task_text)
        _need(type(focused._receipt_json) is str and len(focused._receipt_json) <= 131072,
              "residual_alternatives_focus_receipt_invalid")
        focus_receipt = json.loads(focused._receipt_json)
        _bytes(focus_receipt)
        _need(type(focus_receipt) is dict and focus_receipt.get("schema") == "repository-world-residual-focus@1"
              and focus_receipt.get("baseline_context_cid") == baseline._capsule.capsule_id
              and focus_receipt.get("baseline_prompt_sha256") == baseline_receipt["prompt_sha256"]
              and focus_receipt.get("candidate_context_cid") == candidate._capsule.capsule_id
              and focus_receipt.get("candidate_prompt_sha256") == candidate_receipt["prompt_sha256"],
              "residual_alternatives_focus_baseline_changed")
        _need(_bytes(baseline_receipt["source_binding"]) == _bytes(candidate_receipt["source_binding"]),
              "residual_alternatives_source_binding_changed")
        lineage = focused_recipe["residual_context"]
        binding = replay_repository_residual_context(lineage["record"],
            **_residual_binding_arguments(baseline._capsule, task_id=recipe["task_id"],
                write_paths=recipe["write_paths"], validation_commands=recipe["validation_commands"],
                acceptance_ids=recipe["acceptance_ids"]))
        _need(_bytes(lineage) == _bytes(_residual_recipe(binding)), "residual_alternatives_component_pin_changed")
        compilation_receipt = ContextCompilationReceipt.from_dict(before["compilation_receipt"])
        _need(_contract_bytes(compilation_receipt.to_record()) == _bytes(before["compilation_receipt"]),
              "residual_alternatives_compilation_receipt_changed")
        compilation = ContextCompileResult(baseline._capsule, compilation_receipt, compilation_receipt.decisions)
        # prepare verifies this exact reconstruction with the retained live
        # compiler; recorded bytes alone cannot provide that live admission.
        full = prepare_repository_world_packet(adapter=baseline._adapter, compilation=compilation,
            intent_world_observation=baseline._intent_world_observation,
            residual_packet=ResidualLlmPacket.from_dict(binding.packet), **_arguments(recipe))
        full_receipt = PreparedRepositoryWorldPacket.receipt(full)
        _need(_bytes(full_receipt["residual_context"]) == _bytes(candidate_receipt["residual_context"])
              == _bytes(binding.receipt), "residual_alternatives_failure_changed")
        _need(full._capsule is baseline._capsule, "residual_alternatives_baseline_population_changed")
        receipt = dict(schema=PAIR_SCHEMA,
            baseline_prompt_sha256=baseline_receipt["prompt_sha256"], baseline_prompt_bytes=baseline_receipt["prompt_bytes"],
            full_prompt_sha256=full_receipt["prompt_sha256"], full_prompt_bytes=full_receipt["prompt_bytes"],
            focused_prompt_sha256=candidate_receipt["prompt_sha256"], focused_prompt_bytes=candidate_receipt["prompt_bytes"],
            baseline_context_cid=baseline._capsule.capsule_id, full_context_cid=full._capsule.capsule_id,
            focused_context_cid=candidate._capsule.capsule_id, residual_packet_id=binding.packet_id,
            residual_context_id=binding.context_id, residual_context_sha256=binding.receipt["context_sha256"],
            source_binding=baseline_receipt["source_binding"], task_text_sha256=baseline_receipt["task_text_sha256"],
            required_population_sha256=_sha(_bytes(_population(baseline._capsule, True))),
            full_optional_population_sha256=_sha(_bytes(_population(full._capsule, False))),
            focused_optional_population_sha256=_sha(_bytes(_population(candidate._capsule, False))),
            required_reference_count=len(_required(baseline._capsule)),
            full_optional_count=len(_population(full._capsule, False)),
            focused_optional_count=len(_population(candidate._capsule, False)),
            full_baseline_optional_preserved=True, full_task_invariants_preserved=True,
            identical_residual_context=True, provider_calls=0, provider_input_tokens=None,
            **dict.fromkeys(_FALSE, False))
        result = PreparedRepositoryWorldResidualAlternatives(_SEAL, baseline=baseline, full=full,
            focused=candidate, focus_request=focused, receipt_bytes=_bytes(receipt))
        with _LOCK:
            _ISSUED[result] = PreparedRepositoryWorldResidualAlternatives._seal_material(result)
        validate_repository_world_residual_alternatives(result, workspace=baseline._workspace, task_text=task_text)
        return result


def validate_repository_world_residual_alternatives(alternatives, *, workspace, task_text):
    """Check original custody and complete all native fences after pure work."""
    _need(type(alternatives) is PreparedRepositoryWorldResidualAlternatives,
          "residual_alternatives_exact_type_required")
    PreparedRepositoryWorldResidualAlternatives._retained(alternatives)
    _need(isinstance(workspace, Path) and workspace.is_absolute()
          and workspace == alternatives._baseline._workspace and type(task_text) is str,
          "residual_alternatives_workspace_or_task_invalid")
    receipt = json.loads(alternatives._receipt_bytes)
    _need(_sha(task_text.encode("utf-8", "strict")) == receipt["task_text_sha256"],
          "residual_alternatives_task_changed")
    alternatives._compiler.verify_compile_result(alternatives._focus_request.compilation)
    # Compiler verification can invoke retained estimator/value-policy callbacks.
    # Refuse changes made by those callbacks before the final native closure.
    PreparedRepositoryWorldResidualAlternatives._retained(alternatives)
    # Source/intent checks are last: no count callback, file read, compile,
    # metadata publication or receipt reconstruction follows these fences.
    _current(alternatives._baseline, task_text=task_text)
    _current(alternatives._full, task_text=task_text)
    _current(alternatives._focused, task_text=task_text)
    return receipt


def _pair_receipt(value):
    _need(type(value) is dict and set(value) == _PAIR_FIELDS and value["schema"] == PAIR_SCHEMA,
          "residual_alternatives_receipt_invalid")
    captured = json.loads(_bytes(value))
    for name in ("baseline_prompt", "full_prompt", "focused_prompt"):
        _need(_hash(captured[name + "_sha256"]) and _integer(captured[name + "_bytes"], 1, 262144),
              "residual_alternatives_prompt_pin_invalid")
    for name in ("task_text_sha256", "residual_context_sha256", "required_population_sha256",
                 "full_optional_population_sha256", "focused_optional_population_sha256"):
        _need(_hash(captured[name]), "residual_alternatives_hash_invalid")
    for name in ("baseline_context_cid", "full_context_cid", "focused_context_cid", "residual_packet_id", "residual_context_id"):
        _need(_text(captured[name]), "residual_alternatives_identity_invalid")
    _need(captured["baseline_context_cid"] == captured["full_context_cid"], "residual_alternatives_full_context_changed")
    for name in ("required_reference_count", "full_optional_count", "focused_optional_count"):
        _need(_integer(captured[name], 0, 512), "residual_alternatives_population_invalid")
    _need(all(captured[name] is True for name in ("full_baseline_optional_preserved", "full_task_invariants_preserved", "identical_residual_context"))
          and all(captured[name] is False for name in _FALSE)
          and type(captured["provider_calls"]) is int and captured["provider_calls"] == 0
          and captured["provider_input_tokens"] is None, "residual_alternatives_scope_invalid")
    source = captured["source_binding"]
    _need(type(source) is dict and set(source) == {"generation", "source_tree_sha256", "semantic_state_cid", "semantic_root_cid", "capsule_index_cid"}
          and _integer(source["generation"], 1) and _hash(source["source_tree_sha256"])
          and all(_text(source[name]) for name in ("semantic_state_cid", "semantic_root_cid", "capsule_index_cid")),
          "residual_alternatives_source_invalid")
    return captured


def validate_repository_world_residual_alternatives_receipt(value):
    """Check recorded pair shape; no live issuance or provenance is restored."""
    return _pair_receipt(value)


def _counter_snapshot(counter, provider, model):
    from .router_implementation_runner import PromptTokenCounter
    _need(type(counter) is PromptTokenCounter, "residual_selection_exact_counter_required")
    binding = {name: getattr(counter, name) for name in _COUNTER_FIELDS}
    callback = counter.count_tokens
    _need(all(_text(value) for value in binding.values()) and binding["provider"] == provider
          and binding["model"] == model and callable(callback), "residual_selection_counter_binding_invalid")
    return binding, callback


def _counter_current(counter, snapshot):
    try:
        current = _counter_snapshot(counter, snapshot[0]["provider"], snapshot[0]["model"])
    except Exception:
        raise RepositoryWorldResidualSelectionError("residual_selection_counter_changed") from None
    _need(current[0] == snapshot[0] and current[1] is snapshot[1], "residual_selection_counter_changed")


def _choice(candidates, status):
    if status != "measured":
        return "full", "tokenizer_" + status
    if candidates["focused"]["model_prompt_bytes"] >= candidates["full"]["model_prompt_bytes"]:
        return "full", "focused_not_smaller_in_bytes"
    if candidates["focused"]["input_tokens"] >= candidates["full"]["input_tokens"]:
        return "full", "focused_not_fewer_local_tokens"
    return "focused", "focused_strictly_fewer_bytes_and_local_tokens"


def select_repository_world_residual_prompt(*, alternatives_receipt, full_complete_prompt,
    focused_complete_prompt, provider, model, token_counter, max_input_tokens=None, native_output_schema=None):
    """Select only a strict byte AND local-token reduction of complete input."""
    from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES, PromptTokenCounter
    pair = _pair_receipt(alternatives_receipt)
    _need(_text(provider) and _text(model), "residual_selection_route_invalid")
    _need(max_input_tokens is None or _integer(max_input_tokens, 1), "residual_selection_cap_invalid")
    prompts = dict(full=full_complete_prompt, focused=focused_complete_prompt)
    candidates = {}
    for label, prompt in prompts.items():
        _need(type(prompt) is str and prompt.strip() and len(prompt) <= 2 * MAX_MODEL_PROMPT_BYTES,
              "residual_selection_prompt_invalid")
        raw = prompt.encode("utf-8", "strict")
        _need(len(raw) <= 2 * MAX_MODEL_PROMPT_BYTES, "residual_selection_prompt_bound")
        candidates[label] = dict(model_prompt_sha256=_sha(raw), model_prompt_bytes=len(raw), content_tokens=None, input_tokens=None)
    _need(native_output_schema is None or type(native_output_schema) is dict, "residual_selection_schema_invalid")
    schema = _wire(native_output_schema, maximum=MAX_MODEL_PROMPT_BYTES).decode("utf-8") if native_output_schema is not None else None
    _need(schema is None or len(schema.encode()) <= MAX_MODEL_PROMPT_BYTES, "residual_selection_schema_bound")
    status, error_type, schema_tokens, binding = "unavailable", None, None, None
    if token_counter is not None:
        snapshot = _counter_snapshot(token_counter, provider, model)
        binding = snapshot[0]
        try:
            counts = {}
            for label, prompt in prompts.items():
                count = PromptTokenCounter.count(token_counter, prompt)
                _counter_current(token_counter, snapshot)
                _need(_integer(count, 1), "residual_selection_count_bound")
                counts[label] = count
            schema_tokens = PromptTokenCounter.count(token_counter, schema) if schema is not None else 0
            _counter_current(token_counter, snapshot)
            _need(_integer(schema_tokens, 1 if schema is not None else 0), "residual_selection_count_bound")
            _need(all(count <= MAX_COUNT - schema_tokens for count in counts.values()), "residual_selection_count_overflow")
        except Exception as error:
            # A mutated identity is a refusal even when the callback also raises.
            _counter_current(token_counter, snapshot)
            if isinstance(error, RepositoryWorldResidualSelectionError) and str(error) == "residual_selection_counter_changed":
                raise
            status, schema_tokens = "failed", None
            error_type = type(error).__name__ if type(error).__name__ in {"ValueError", "TypeError", "RuntimeError", "OverflowError"} else "other"
        else:
            status = "measured"
            for label, count in counts.items():
                candidates[label].update(content_tokens=count, input_tokens=count + schema_tokens)
    _need(max_input_tokens is None or status == "measured", "residual_selection_cap_requires_counter")
    selected, reason = _choice(candidates, status)
    chosen = candidates[selected]
    _need(chosen["model_prompt_bytes"] <= MAX_MODEL_PROMPT_BYTES, "residual_selection_selected_prompt_bound")
    _need(max_input_tokens is None or chosen["input_tokens"] <= max_input_tokens, "residual_selection_selected_cap_exceeded")
    receipt = dict(schema=SELECTION_SCHEMA, alternatives_receipt_sha256=_sha(_bytes(pair)),
        provider=provider, model=model, selected_representation=selected, selection_reason=reason,
        candidates=candidates, measurement_status=status, measurement_error_type=error_type,
        tokenizer_binding=binding, count_scope="rendered_user_prompt_and_separate_native_output_schema",
        native_output_schema_tokens=schema_tokens, native_output_schema_sha256=_sha(schema.encode()) if schema is not None else None,
        selected_content_tokens=chosen["content_tokens"], selected_input_tokens=chosen["input_tokens"],
        max_input_tokens=max_input_tokens, max_prompt_bytes=MAX_MODEL_PROMPT_BYTES,
        provider_framing_measured=False, provider_tokenizer_equivalence_verified=False,
        total_session_token_savings=None, semantic_equivalence_verified=False, omission_authority=False, completion_authority=False)
    return selected, receipt, repository_world_residual_selected_measurement(receipt)


def validate_repository_world_residual_selection_receipt(value):
    """Validate recorded scope/arithmetic; this does not rerun a trusted counter."""
    from .router_implementation_runner import MAX_MODEL_PROMPT_BYTES
    _need(type(value) is dict and set(value) == _SELECTION_FIELDS and value["schema"] == SELECTION_SCHEMA,
          "residual_selection_receipt_invalid")
    result = json.loads(_bytes(value))
    _need(_hash(result["alternatives_receipt_sha256"]) and _text(result["provider"]) and _text(result["model"]), "residual_selection_receipt_route_invalid")
    status = result["measurement_status"]
    _need(status in ("measured", "failed", "unavailable"), "residual_selection_status_invalid")
    _need(result["measurement_error_type"] is None if status != "failed" else
          type(result["measurement_error_type"]) is str and result["measurement_error_type"] in
          {"ValueError", "TypeError", "RuntimeError", "OverflowError", "other"}, "residual_selection_error_invalid")
    binding = result["tokenizer_binding"]
    _need(binding is None and status == "unavailable" or type(binding) is dict and set(binding) == set(_COUNTER_FIELDS)
          and all(_text(part) for part in binding.values()) and binding["provider"] == result["provider"]
          and binding["model"] == result["model"] and status != "unavailable", "residual_selection_recorded_counter_invalid")
    schema_hash, schema_tokens = result["native_output_schema_sha256"], result["native_output_schema_tokens"]
    _need(schema_hash is None or _hash(schema_hash), "residual_selection_schema_pin_invalid")
    _need(_integer(schema_tokens, 1 if schema_hash is not None else 0) if status == "measured" else schema_tokens is None,
          "residual_selection_schema_count_invalid")
    _need(status != "measured" or schema_hash is not None or schema_tokens == 0, "residual_selection_absent_schema_count_invalid")
    candidates = result["candidates"]
    _need(type(candidates) is dict and set(candidates) == {"full", "focused"}, "residual_selection_candidates_invalid")
    for candidate in candidates.values():
        _need(type(candidate) is dict and set(candidate) == {"model_prompt_sha256", "model_prompt_bytes", "content_tokens", "input_tokens"}
              and _hash(candidate["model_prompt_sha256"]) and _integer(candidate["model_prompt_bytes"], 1, 2 * MAX_MODEL_PROMPT_BYTES),
              "residual_selection_candidate_pin_invalid")
        if status == "measured":
            _need(_integer(candidate["content_tokens"], 1) and _integer(candidate["input_tokens"], 1)
                  and candidate["content_tokens"] + schema_tokens == candidate["input_tokens"], "residual_selection_arithmetic_invalid")
        else:
            _need(candidate["content_tokens"] is None and candidate["input_tokens"] is None, "residual_selection_partial_counts_forbidden")
    selected, reason = _choice(candidates, status)
    _need(result["selected_representation"] == selected and result["selection_reason"] == reason, "residual_selection_choice_invalid")
    chosen = candidates[selected]
    _need(result["selected_content_tokens"] == chosen["content_tokens"]
          and type(result["selected_content_tokens"]) is type(chosen["content_tokens"])
          and result["selected_input_tokens"] == chosen["input_tokens"]
          and type(result["selected_input_tokens"]) is type(chosen["input_tokens"]), "residual_selection_selected_counts_invalid")
    _need(type(result["max_prompt_bytes"]) is int and result["max_prompt_bytes"] == MAX_MODEL_PROMPT_BYTES
          and chosen["model_prompt_bytes"] <= MAX_MODEL_PROMPT_BYTES, "residual_selection_recorded_byte_cap_invalid")
    cap = result["max_input_tokens"]
    _need(cap is None or status == "measured" and _integer(cap, 1) and chosen["input_tokens"] <= cap, "residual_selection_recorded_cap_invalid")
    _need(result["count_scope"] == "rendered_user_prompt_and_separate_native_output_schema"
          and result["total_session_token_savings"] is None and all(result[name] is False for name in
          ("provider_framing_measured", "provider_tokenizer_equivalence_verified", "semantic_equivalence_verified", "omission_authority", "completion_authority")),
          "residual_selection_recorded_scope_invalid")
    return result


def repository_world_residual_selected_measurement(value):
    selection = validate_repository_world_residual_selection_receipt(value)
    candidate = selection["candidates"][selection["selected_representation"]]
    names = ("measurement_status", "measurement_error_type", "tokenizer_binding", "count_scope",
        "native_output_schema_tokens", "native_output_schema_sha256", "selected_content_tokens", "selected_input_tokens",
        "max_input_tokens", "max_prompt_bytes", "provider_framing_measured", "provider_tokenizer_equivalence_verified",
        "total_session_token_savings", "completion_authority")
    return dict(schema="supervisor-world-task-prompt-measurement@1", measurement_kind="declared_local_tokenizer",
        model_prompt_sha256=candidate["model_prompt_sha256"], model_prompt_bytes=candidate["model_prompt_bytes"],
        **{name: selection[name] for name in names}, provider_system_prompt_measured=False,
        provider_tool_schemas_measured=False, provider_internal_history_measured=False)
