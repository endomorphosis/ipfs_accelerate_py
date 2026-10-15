"""Replay retained semantic dialogue inputs without exporting their contents.

Historical producer checks authenticate retained artifacts, not current source.
Recorded local counts and provider delivery observations remain separate from
exact reconstructed bytes. This module invokes no provider or tokenizer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


SCHEMA = "terminal-semantic-decision-dialogue-input-audit@1"
MAX_INPUT_BYTES = 256_000
MAX_DIALOGUE_BYTES = 300_000
MAX_RESULT_BYTES = 32_768
MAX_ID_BYTES = 512
MAX_INTEGER = 2**63 - 1


class _Refusal(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Refusal(reason)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _text(value, maximum=MAX_ID_BYTES):
    return (type(value) is str and bool(value) and value == value.strip()
            and len(value.encode()) <= maximum
            and not any(char in value for char in ("\x00", "\n", "\r")))


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value, minimum=0, maximum=MAX_INTEGER):
    return type(value) is int and minimum <= value <= maximum


def _path(value):
    _require(_text(value, 4096), "invalid_path_identity")
    path = Path(value)
    _require(path.is_absolute() and str(path) == value
             and ".." not in path.parts and "." not in value.split("/"),
             "invalid_path_identity")
    return path


def _base():
    return {
        "schema": SCHEMA, "status": "unverified", "reason": "input_unavailable",
        "original_native_input_verified": False,
        "all_recorded_round_inputs_verified": False,
        "recorded_round_count": 0, "verified_round_count": 0, "attempts": [],
        "recorded_count_algebra_consistent": False,
        "recorded_local_input_comparison": None,
        "provider_calls": 0, "freshness_revalidated": False,
        "completion_authority": False, "execution_authority": False,
        "token_savings_established": False, "raw_prompts_exported": False,
        "input_scope": "complete rendered router user messages; provider internal context is outside scope",
        "provider_delivery_scope": "retained provider receipts; reconstruction alone does not establish dispatch",
    }


def _policy(*, launch, workspace, native_prompt, repository):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
    from ipfs_accelerate_py.agent_supervisor.runtime.local_prompt_tokenizer import (
        _bounded_local_bytes, _json_object, _local_path,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import SemanticDecisionPolicy
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_projection import (
        SemanticProjectionBindings, _encoded, _structure,
    )

    _require(type(launch) is dict, "missing_worker_launch")
    path = _local_path(_path(launch.get("launch_policy_artifact")))
    digest = launch.get("launch_policy_sha256")
    _require(_digest(digest) and not path.is_relative_to(workspace), "invalid_launch_policy_binding")
    raw = _bounded_local_bytes(path, maximum=MAX_INPUT_BYTES, expected_sha256=digest)
    descriptor = _json_object(raw, "historical semantic decision launch policy")
    _structure(descriptor, MAX_INPUT_BYTES)
    retrieval_profile = descriptor.get("schema") == "supervisor-semantic-decision-launch-policy@3"
    fields = {"schema", "task_id", "parent_artifact", "parent_sha256", "policy"}
    if retrieval_profile:
        fields.add("retrieval_query_binding")
    _require(set(descriptor) == fields
             and descriptor.get("schema") in {"supervisor-semantic-decision-launch-policy@1",
                                               "supervisor-semantic-decision-launch-policy@2",
                                               "supervisor-semantic-decision-launch-policy@3"},
             "invalid_launch_policy_descriptor")
    value = descriptor["policy"]
    policy = SemanticDecisionPolicy.from_dict(value)
    _require((retrieval_profile and policy.schema_revision == "2" and policy.task_text_source_path is not None)
             or (not retrieval_profile and descriptor["schema"] ==
                 "supervisor-semantic-decision-launch-policy@" + policy.schema_revision),
             "invalid_selection_policy_revision")
    retrieval_binding = None
    if retrieval_profile:
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_retrieval_query_reference import (
            SemanticRetrievalQueryBinding,
        )
        retrieval_binding = SemanticRetrievalQueryBinding.from_dict(descriptor["retrieval_query_binding"])
        _require(retrieval_binding.task_id == descriptor["task_id"], "retrieval_policy_task_differs")
        _require(launch.get("retrieval_query_binding") == retrieval_binding.to_dict(),
                 "retrieval_launch_binding_differs")
    else:
        _require("retrieval_query_binding" not in launch, "unregistered_retrieval_launch_binding")
    wire, suffix, positions, references, text, artifact = codec._prompt_parts(native_prompt)
    _require(descriptor["task_id"] == wire["objective_id"]
             and descriptor["parent_artifact"] == artifact
             and descriptor["parent_sha256"] == _sha(text.encode()), "policy_parent_binding_differs")
    semantic, _ = codec._verified_semantic(repository, artifact=artifact, text=text,
                                          task_id=wire["objective_id"], current=False)
    bindings = SemanticProjectionBindings(
        artifact=artifact, expected_sha256=_sha(text.encode()), task_id=wire["objective_id"],
        semantic_root_cid=semantic["semantic_root_cid"], scope_cid=semantic["scope_cid"],
        repository_id=wire["repository_id"], tree_id=wire["tree_id"],
    )
    result = (policy, bindings, wire, suffix, positions, references, text, _sha(_encoded(policy.to_dict())))
    return result if retrieval_binding is None else (*result, retrieval_binding)


def _suffix(*, launch, native_prompt, repository):
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_residual_context import load_doctor_residual_advisory
    from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import load_public_instruction

    suffix = ""
    for name, loader in (("doctor_residual_context", load_doctor_residual_advisory),
                         ("public_instruction", load_public_instruction)):
        retained = launch.get(name)
        if retained is None:
            continue
        _require(type(retained) is dict, "invalid_required_suffix_binding")
        _require(_digest(retained.get("artifact_sha256")) and _text(retained.get("task_cid")),
                 "invalid_required_suffix_binding")
        block, observed = loader(
            artifact=_path(retained.get("artifact")), expected_sha256=retained["artifact_sha256"],
            repository=repository, task_cid=retained["task_cid"], prompt=native_prompt,
            require_current_source=False,
        )
        _require(retained.get("source_freshness_verified") is True
                 and retained.get("historical_replay") is False,
                 "invalid_required_suffix_history")
        _require(all(retained.get(key) == value for key, value in observed.items()
                     if key not in {"source_freshness_verified", "historical_replay"}),
                 "required_suffix_receipt_differs")
        suffix += block
    _require(launch.get("required_suffix_sha256") == _sha(suffix.encode())
             and type(launch.get("required_suffix_bytes")) is int
             and launch["required_suffix_bytes"] == len(suffix.encode()), "required_suffix_binding_differs")
    return suffix


def _packet(*, wire, suffix, positions, references, text, policy, bindings,
            retrieval_query_binding=None, repository=None):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import (
        build_semantic_decision_packet, semantic_decision_packet_references,
    )

    view = build_semantic_decision_packet(original_text=text, bindings=bindings, policy=policy)
    coverage = tuple(sorted({item for reference in references for item in reference.coverage_ids}))
    projected = semantic_decision_packet_references(
        view=view, original_text=text, bindings=bindings, policy=policy, coverage_ids=coverage,
        chunk_bytes=6144,
    )
    _require(bool(projected) and all(reference.required for reference in projected)
             and set(coverage) <= {item for reference in projected for item in reference.coverage_ids},
             "packet_required_coverage_differs")
    retrieval_positions, retrieval_projected, retrieval_held = (), (), None
    if retrieval_query_binding is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_projection import _encoded
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_retrieval_query_reference import (
            load_historical_retrieval_query_context,
            validate_native_retrieval_references,
            build_retrieval_query_reference,
            restore_retrieval_query_reference,
            retrieval_query_reference_references,
        )
        native_text = load_historical_retrieval_query_context(
            repository=repository, binding=retrieval_query_binding,
            semantic_original_text=text, semantic_bindings=bindings)
        retrieval_positions, native_references = validate_native_retrieval_references(
            wire=wire, binding=retrieval_query_binding, native_text=native_text)
        retrieval_view = build_retrieval_query_reference(
            native_text=native_text, binding=retrieval_query_binding, semantic_view=view,
            semantic_original_text=text, semantic_bindings=bindings, policy=policy)
        restored = restore_retrieval_query_reference(
            view=retrieval_view, native_text=native_text, binding=retrieval_query_binding,
            semantic_view=view, semantic_original_text=text, semantic_bindings=bindings, policy=policy)
        _require(restored == native_text, "historical_retrieval_inverse_differs")
        retrieval_coverage = tuple(sorted({item for reference in native_references
                                          for item in reference.coverage_ids}))
        retrieval_projected = retrieval_query_reference_references(
            view=retrieval_view, native_text=native_text, binding=retrieval_query_binding,
            semantic_view=view, semantic_original_text=text, semantic_bindings=bindings,
            policy=policy, coverage_ids=retrieval_coverage, chunk_bytes=6144)
        _require(bool(retrieval_projected) and all(reference.required for reference in retrieval_projected)
                 and set(retrieval_coverage) <= {item for reference in retrieval_projected
                                                for item in reference.coverage_ids}
                 and not set(positions).intersection(retrieval_positions),
                 "retrieval_required_coverage_differs")
        retrieval_held = {
            "binding": retrieval_query_binding.to_dict(), "native_context_sha256": _sha(native_text.encode()),
            "view_sha256": _sha(_encoded(retrieval_view)),
            "native_reference_records_sha256": _sha(_encoded([reference.to_record() for reference in native_references])),
        }
    evidence = []
    for index, reference in enumerate(wire["evidence"]):
        if index == positions[0]:
            evidence.extend(item.to_record() for item in projected)
        if retrieval_positions and index == retrieval_positions[0]:
            evidence.extend(item.to_record() for item in retrieval_projected)
        if index not in set(positions) | set(retrieval_positions):
            evidence.append(reference)
    if retrieval_query_binding is not None:
        reference_ids = [reference["reference_id"] for reference in evidence]
        _require(len(reference_ids) == len(set(reference_ids)), "retrieval_projected_reference_ids_duplicate")
    prompt = codec._json({**wire, "evidence": evidence}) + suffix
    _require(len(prompt.encode()) <= codec.MAX_BYTES, "packet_input_exceeds_bound")
    return (prompt, view) if retrieval_held is None else (prompt, view, retrieval_held)


def _expansions(*, rows, attempts, max_calls, view, policy, bindings, text, repository):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_projection import _encoded
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_packet import expand_semantic_decision_packet

    _require(type(rows) is list and len(rows) <= 3, "invalid_expansion_population")
    deliveries, seen, previous = [], set(), -1

    def historical_parent(*, artifact, expected_sha256, task_id):
        raw = codec._read(repository, artifact)
        _require(_sha(raw) == expected_sha256 and raw == text.encode(), "historical_parent_changed")
        codec._verified_semantic(repository, artifact=artifact, text=raw.decode(), task_id=task_id, current=False)
        return raw.decode()

    for row in rows:
        _require(type(row) is dict, "invalid_expansion_record")
        after, identities = row.get("after_generation"), row.get("capsule_ids")
        _require(_integer(after, 0, 2) and previous < after < len(attempts) and after + 1 < max_calls,
                 "invalid_expansion_generation")
        _require(type(identities) is list and 1 <= len(identities) <= 4
                 and all(_text(identity) for identity in identities)
                 and len(set(identities)) == len(identities) and not seen.intersection(identities),
                 "invalid_expansion_subset")
        expected_calls = [attempt["call_id"] for attempt in attempts if attempt["generation"] > after]
        _require(row.get("included_in_invocation_attempts") == expected_calls,
                 "expansion_delivery_schedule_differs")
        handles = view["deferred_references"]
        _require(len(handles) == 1, "missing_expansion_archive")
        # The descriptor always contains token_count=0. This structural sentinel
        # satisfies the builder's gate without measuring or asserting tokens.
        expanded = expand_semantic_decision_packet(
            view=view, reference_id=handles[0]["reference_id"], bindings=bindings, policy=policy,
            resolve_original=historical_parent, count_tokens=lambda _: 1,
            max_bytes=65536, max_tokens=128000, capsule_ids=tuple(identities),
        )
        encoded = _encoded(expanded.to_dict())
        _require(row.get("referenced_content_id") == expanded.referenced_content_id
                 and row.get("descriptor_sha256") == _sha(encoded)
                 and type(row.get("descriptor_bytes")) is int and row["descriptor_bytes"] == len(encoded),
                 "expansion_descriptor_differs")
        deliveries.append((after, expanded))
        seen.update(identities)
        previous = after
    _require(set(range(max(0, len(attempts) - 1))) <= {after for after, _ in deliveries},
             "missing_continuation_expansion")
    return deliveries


def _delivery(*, attempt, dialogue, binding, prompt, model_prompt, advisory):
    result = {"provider_dispatch_reported": None, "provider_receipt_binding_validated": False,
              "native_receipt_input_binding_validated": False}
    receipt = attempt.get("provider_receipt")
    if type(receipt) is not dict:
        return result
    dispatched = receipt.get("provider_called")
    if type(dispatched) is bool:
        result["provider_dispatch_reported"] = dispatched
    expected = {
        "schema": "router-semantic-decision-provider@1", "call_id": attempt["call_id"],
        "session_id": dialogue["session_id"], "generation": attempt["generation"],
        "binding_sha256": binding, "provider": dialogue["provider"], "model": dialogue["model"],
        "workspace": dialogue["workspace"], "prompt_sha256": _sha(prompt.encode()),
        "prompt_bytes": len(prompt.encode()), "model_prompt_sha256": _sha(model_prompt.encode()),
        "model_prompt_bytes": len(model_prompt.encode()),
    }
    if attempt.get("provider_receipt_bound") is not True or not all(
            type(receipt.get(key)) is type(value) and receipt.get(key) == value for key, value in expected.items()):
        return result
    result["provider_receipt_binding_validated"] = True
    native = receipt.get("native_invocation")
    if dialogue["provider"] == "grok_cli":
        return result
    if type(native) is not dict or not _text(native.get("invocation_id")):
        return result
    expected_native = {
        "schema": "router-implementation-invocation@1", "provider": dialogue["provider"],
        "model": dialogue["model"], "workspace": dialogue["workspace"], "purpose": "coding",
        "prompt_sha256": _sha(prompt.encode()), "prompt_bytes": len(prompt.encode()),
        "native_prompt_sha256": _sha(prompt.encode()), "native_prompt_bytes": len(prompt.encode()),
        "router_prompt_sha256": _sha(prompt.encode()), "router_prompt_bytes": len(prompt.encode()),
        "model_prompt_sha256": _sha(model_prompt.encode()), "model_prompt_bytes": len(model_prompt.encode()),
        "workspace_advisory_sha256": _sha(advisory.encode()), "workspace_advisory_bytes": len(advisory.encode()),
    }
    if dispatched is True and all(type(native.get(key)) is type(value) and native.get(key) == value
                                  for key, value in expected_native.items()):
        result.update(native_invocation_id_sha256=_sha(native["invocation_id"].encode()),
                      native_receipt_input_binding_validated=True)
    return result


def _counts(*, result, dialogue, native_model, first_model):
    binding = dialogue.get("tokenizer_binding")
    kind, basis = dialogue.get("token_count_kind"), dialogue.get("tokenizer_compatibility_basis")
    counts = [attempt.get("content_tokens") for attempt in dialogue["attempts"]]
    baseline, first, total = (dialogue.get(name) for name in (
        "native_content_tokens", "packet_first_content_tokens", "local_input_content_token_sum"))
    valid = (type(binding) is dict and set(binding) == {
        "provider", "model", "tokenizer_id", "tokenizer_revision", "special_token_policy"}
        and all(_text(value) for value in binding.values())
        and binding["provider"] == dialogue["provider"] and binding["model"] == dialogue["model"]
        and type(kind) is str and kind in {"tokenizer", "estimate"} and _text(basis)
        and all(_integer(value, 1) for value in [baseline, first, *counts])
        and _integer(total, 1) and total == sum(counts)
        and all(attempt.get("content_token_count_kind") == kind for attempt in dialogue["attempts"]))
    if valid and counts:
        native_selected = dialogue["selected_representation"] == "native"
        valid = counts[0] == (baseline if native_selected else first)
        native_fits = len(native_model.encode()) <= MAX_INPUT_BYTES
        for name in ("max_input_tokens", "max_total_input_tokens"):
            limit = dialogue.get(name)
            if limit is not None and not _integer(limit, 1):
                valid = False
            elif limit is not None and baseline > limit:
                native_fits = False
            if limit is not None and _integer(limit, 1):
                if name == "max_input_tokens" and any(amount > limit for amount in counts):
                    valid = False
                if name == "max_total_input_tokens" and total > limit:
                    valid = False
        valid = valid and native_selected == (native_fits and baseline <= first)
        known = {_sha(native_model.encode()): baseline, _sha(first_model.encode()): first}
        if _sha(native_model.encode()) == _sha(first_model.encode()) and baseline != first:
            valid = False
        for attempt in dialogue["attempts"]:
            digest, amount = attempt["model_prompt_sha256"], attempt["content_tokens"]
            if digest in known and known[digest] != amount:
                valid = False
            known[digest] = amount
    result["recorded_count_algebra_consistent"] = bool(valid)
    result["recorded_count_reason"] = ("historical_local_count_algebra_matches" if valid
                                       else "historical_local_count_algebra_unavailable_or_conflicting")
    if valid and result["all_recorded_round_inputs_verified"] and counts:
        result["recorded_local_input_comparison"] = {
            "native_content_tokens": baseline, "all_round_content_token_sum": total,
            "difference_from_single_native_input": total - baseline,
            "count_kind": kind,
            "tokenizer_compatibility_basis": (basis if basis in {
                "owner_declaration", "caller_declaration", "installed_tiktoken_model_mapping",
                "offline_o200k_base_proxy", "utf8_byte_estimate", "authored_fixture_only",
            } else "other_recorded_basis"),
            "tokenizer_compatibility_basis_sha256": _sha(basis.encode()),
            "tokenizer_binding_sha256": _sha(json.dumps(binding, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()),
            "historical_counts_recomputed": False, "provider_usage": False,
            "comparison_scope": "recorded complete user inputs versus one recorded native user input; no task-success equivalence",
        }


def audit_semantic_decision_dialogue(*, native_prompt: str, dialogue: dict,
                                   repository: Path, workspace_root: Path) -> dict:
    """Reconstruct at most four retained user inputs; ordinary failures abstain."""
    result = _base()
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_projection import _encoded, _structure
        from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_dialogue import _round_prompt
        from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt

        _require(type(native_prompt) is str and 0 < len(native_prompt.encode()) <= MAX_INPUT_BYTES,
                 "invalid_original_native_input")
        _structure(dialogue, MAX_DIALOGUE_BYTES)
        _require(type(dialogue) is dict and len(_encoded(dialogue)) <= MAX_DIALOGUE_BYTES
                 and dialogue.get("schema") == "supervisor-semantic-decision-dialogue@1",
                 "invalid_dialogue_record")
        _require(dialogue.get("provider") in {"codex_cli", "grok_cli"}
                 and _text(dialogue.get("model")) and _text(dialogue.get("session_id")),
                 "invalid_dialogue_identity")
        root, allocation = _path(str(repository)), _path(str(workspace_root))
        _require(root.resolve(strict=True) == root and root.is_dir(), "invalid_artifact_repository")
        workspace = _path(dialogue.get("workspace"))
        _require(workspace != allocation and workspace.is_relative_to(allocation)
                 and root != workspace and not root.is_relative_to(workspace), "invalid_allocated_workspace")
        result.update(workspace_sha256=_sha(str(workspace).encode()), provider=dialogue["provider"],
                      model=dialogue["model"], session_id_sha256=_sha(dialogue["session_id"].encode()))
        launch = dialogue.get("worker_launch")
        _require(type(launch) is dict
                 and launch.get("original_native_prompt_sha256") == _sha(native_prompt.encode())
                 and type(launch.get("original_native_prompt_bytes")) is int
                 and launch["original_native_prompt_bytes"] == len(native_prompt.encode()),
                 "original_native_input_binding_differs")
        policy_result = _policy(
            launch=launch, workspace=workspace, native_prompt=native_prompt, repository=root,
        )
        policy, bindings, wire, original_suffix, positions, references, text, policy_sha = policy_result[:8]
        retrieval_binding = policy_result[8] if len(policy_result) == 9 else None
        _require(dialogue.get("task_id") == bindings.task_id
                 and dialogue.get("parent_sha256") == bindings.expected_sha256
                 and dialogue.get("policy_sha256") == policy_sha, "dialogue_parent_or_policy_differs")
        prepared_suffix = _suffix(launch=launch, native_prompt=native_prompt, repository=root)
        prepared = native_prompt + prepared_suffix
        _require(len(prepared.encode()) <= MAX_INPUT_BYTES
                 and dialogue.get("native_prompt_sha256") == _sha(prepared.encode()),
                 "prepared_native_input_binding_differs")
        packet_options = {}
        if retrieval_binding is not None:
            packet_options.update(retrieval_query_binding=retrieval_binding, repository=root)
        packet_result = _packet(wire=wire, suffix=original_suffix + prepared_suffix,
                                positions=positions, references=references, text=text, policy=policy,
                                bindings=bindings, **packet_options)
        projected, view = packet_result[:2]
        round_options = {}
        if retrieval_binding is not None:
            held = packet_result[2]
            _require(dialogue.get("retrieval_query_reference") == held, "retrieval_round_binding_differs")
            round_options["retrieval_query_reference"] = held
        else:
            _require("retrieval_query_reference" not in dialogue, "unregistered_retrieval_round_binding")
        _require(dialogue.get("packet_prompt_sha256") == _sha(projected.encode()), "packet_input_binding_differs")
        result["original_native_input_verified"] = True
        attempts = dialogue.get("attempts")
        _require(type(attempts) is list and len(attempts) <= 4
                 and _integer(dialogue.get("max_calls"), 1, 4) and len(attempts) <= dialogue["max_calls"]
                 and type(dialogue.get("invocation_attempt_count")) is int
                 and dialogue["invocation_attempt_count"] == len(attempts), "invalid_round_population")
        result["recorded_round_count"] = len(attempts)
        seen = set()
        for generation, attempt in enumerate(attempts):
            _require(type(attempt) is dict and _text(attempt.get("call_id"))
                     and attempt["call_id"] not in seen and type(attempt.get("generation")) is int
                     and attempt["generation"] == generation, "invalid_round_identity_or_generation")
            seen.add(attempt["call_id"])
        selected = dialogue.get("selected_representation")
        _require(selected in {"native", "decision_packet"}, "invalid_selected_representation")
        _require(selected != "native" or (len(attempts) <= 1 and dialogue.get("expansions") == []),
                 "invalid_native_fallback_rounds")
        deliveries = _expansions(rows=dialogue.get("expansions"), attempts=attempts, view=view,
                                 max_calls=dialogue["max_calls"], policy=policy, bindings=bindings,
                                 text=text, repository=root)
        first, _ = _round_prompt(projected, dialogue["session_id"], 0, bindings, policy,
                                 dialogue["provider"], dialogue["model"], [], **round_options)
        native_model, _ = render_model_prompt(prompt=prepared, purpose="coding", workspace=workspace)
        first_model, _ = render_model_prompt(prompt=first, purpose="coding", workspace=workspace)
        for attempt in attempts:
            generation = attempt["generation"]
            prompt, binding = _round_prompt(projected, dialogue["session_id"], generation, bindings,
                                            policy, dialogue["provider"], dialogue["model"],
                                            [reference for after, reference in deliveries if after < generation],
                                            **round_options)
            if selected == "native":
                prompt = prepared
            model, advisory = render_model_prompt(prompt=prompt, purpose="coding", workspace=workspace)
            observation = {"call_id_sha256": _sha(attempt["call_id"].encode()), "generation": generation,
                           "status": "unverified", "reason": "recorded_round_input_differs",
                           "model_input_verified": False, "provider_dispatch_reported": None,
                           "provider_receipt_binding_validated": False,
                           "native_receipt_input_binding_validated": False,
                           "historical_count_only": True}
            if attempt.get("status") in ("started", "provider_returned", "provider_failed", "rejected"):
                observation["recorded_attempt_status"] = attempt["status"]
            kind, amount = attempt.get("content_token_count_kind"), attempt.get("content_tokens")
            if type(kind) is str and kind in {"tokenizer", "estimate"} and _integer(amount, 1):
                observation.update(recorded_content_tokens=amount, count_kind=kind)
            expected = {"prompt_sha256": _sha(prompt.encode()), "prompt_bytes": len(prompt.encode()),
                        "model_prompt_sha256": _sha(model.encode()), "model_prompt_bytes": len(model.encode())}
            if (len(model.encode()) <= MAX_INPUT_BYTES and all(
                    type(attempt.get(key)) is type(value) and attempt.get(key) == value
                    for key, value in expected.items())):
                observation.update(expected, status="verified", model_input_verified=True,
                                   reason="exact_historical_round_input_matches")
                observation.update(_delivery(attempt=attempt, dialogue=dialogue, binding=binding,
                                             prompt=prompt, model_prompt=model, advisory=advisory))
                result["verified_round_count"] += 1
            result["attempts"].append(observation)
        result["all_recorded_round_inputs_verified"] = bool(attempts) and result["verified_round_count"] == len(attempts)
        result["status"] = ("verified" if result["all_recorded_round_inputs_verified"] else
                            "partial" if result["verified_round_count"] else "unverified")
        result["reason"] = "per_round_historical_input_comparison"
        _counts(result=result, dialogue=dialogue, native_model=native_model, first_model=first_model)
    except _Refusal as error:
        result.update(status="partial" if result["verified_round_count"] else "unverified", reason=str(error))
    except Exception:
        result.update(status="partial" if result["verified_round_count"] else "unverified",
                      reason="historical_artifact_or_record_unavailable")
    if len(json.dumps(result, sort_keys=True).encode()) > MAX_RESULT_BYTES:
        return _base()
    return result


def audit_semantic_typed_nomination_lineage(
    *, native_prompt: str, owner_binding: dict, dialogue: dict, repository: Path,
    workspace_root: Path, expected_origin_nomination_sha256: str, expected_task_id: str,
) -> dict:
    """Verify one owner lineage against a reconstructed original native input.

    The caller supplies the prompt rendered from a trusted retained context
    capsule, before worker suffixes. Failure raises so the caller can retain
    usage observations while withholding coverage. No printed marker is read.
    """
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_decision_usage import (
        MAX_OWNER_BINDING_BYTES, _dialogue_owner_key, _owner_binding_key,
        _typed_verification_matches_owner, _valid_owner_binding,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_typed_nomination_lineage import (
        load_and_verify_typed_nomination_lineage,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_projection import _structure

    _require(type(native_prompt) is str and 0 < len(native_prompt.encode()) <= MAX_INPUT_BYTES,
             "invalid_original_native_input")
    _structure(owner_binding, MAX_OWNER_BINDING_BYTES)
    _require(_valid_owner_binding(owner_binding) and "typed_nomination" in owner_binding,
             "invalid_typed_owner_binding")
    _require(_digest(expected_origin_nomination_sha256) and _text(expected_task_id, 512)
             and owner_binding["nomination_sha256"] == expected_origin_nomination_sha256
             and owner_binding["task_id"] == expected_task_id,
             "typed_nomination_independent_pin_differs")
    _require(owner_binding["native_prompt_sha256"] == _sha(native_prompt.encode())
             and owner_binding["native_prompt_bytes"] == len(native_prompt.encode()),
             "typed_nomination_original_input_differs")
    _require(type(dialogue) is dict
             and dialogue.get("schema") == "supervisor-semantic-decision-dialogue@1"
             and _dialogue_owner_key(dialogue) == _owner_binding_key(owner_binding),
             "typed_nomination_dialogue_owner_differs")
    root, allocation = _path(str(repository)), _path(str(workspace_root))
    _require(root.resolve(strict=True) == root and root.is_dir(), "invalid_artifact_repository")
    workspace = _path(dialogue.get("workspace"))
    _require(workspace != allocation and workspace.is_relative_to(allocation)
             and root != workspace and not root.is_relative_to(workspace), "invalid_allocated_workspace")
    binding = owner_binding["typed_nomination"]
    result = load_and_verify_typed_nomination_lineage(
        artifact=_path(binding["lineage_artifact"]), expected_sha256=binding["lineage_sha256"],
        native_prompt=native_prompt, repository=root, workspace=workspace,
        expected_origin_nomination_sha256=expected_origin_nomination_sha256,
        expected_task_id=expected_task_id, owner_binding=owner_binding,
        require_current_source=False,
    )
    _require(_typed_verification_matches_owner(result, owner_binding),
             "typed_nomination_verification_binding_differs")
    return dict(result)
