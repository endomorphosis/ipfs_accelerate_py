"""Compare retained native Harbor receipts without reading model transcripts.

Counts retain each provider's native scope and cache semantics. Codex cumulative
session counters require native completion; Grok final-envelope counters also
require explicit usage completeness. Neither establishes complete billing.
Missing or interrupted usage retains observed counters separately from totals.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re

from benchmarks.agent_supervisor.container_coding.benchmark_controls import compare_controls
from benchmarks.agent_supervisor.container_coding import native_usage_accounting

BASELINE = "native-codex-harbor-baseline-receipt@1"
SUPERVISOR = "terminal-full-supervisor-receipt@1"
TOKENS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")
SUPERVISOR_PHASES = ("prepare_seconds", "initial_context_seconds", "planning_seconds",
                     "context_seconds", "doctor_seconds", "post_publication_refresh_seconds")
EMBEDDING_COUNTERS = ("local_embedding_calls", "local_embedding_texts",
                      "remote_embedding_calls", "text_generation_calls")
LABELS = {
    "native-codex": "Native Codex harness",
    "no-index": "Supervisor: no indexed context",
    "full": "Supervisor: full indexed context",
}


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _tokens(value):
    value = value if isinstance(value, dict) else {}
    return {
        key: value.get(key) if type(value.get(key)) is int and value[key] >= 0 else None
        for key in TOKENS
    }


def _text(value):
    return value[:256] if isinstance(value, str) else None


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _boolean(value):
    return value if type(value) is bool else None


def _completion(values):
    values = list(values)
    if any(value is False for value in values):
        return False
    return True if values and all(value is True for value in values) else None


def _baseline_completion(native):
    sessions = native.get("sessions")
    flags = [_boolean(row.get("task_complete_observed"))
             if isinstance(row, dict) else None for row in sessions] if isinstance(sessions, list) else []
    explicit = _boolean(native.get("task_complete_observed"))
    if explicit is False or any(flag is False for flag in flags):
        return False
    return True if explicit is True else _completion(flags)


def _archive_digest(raw):
    value = _mapping(_mapping(raw.get("agent_context")).get("metadata")).get("runtime_archive_sha256")
    return value if isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value) else None


def _semantic_metadata_observation(value):
    """Project declared representation counts, never usage or tokenizer authority."""
    unavailable = {"status": "unavailable", "provider_usage_reported": False,
        "tokenizer_compatibility_verified": False, "complete_run_token_savings_claimed": False}
    if type(value) is not dict:
        return unavailable
    schema = value.get("schema")
    if type(schema) is not str or schema not in {"supervisor-semantic-metadata-selection@1", "supervisor-semantic-metadata-selection@2"}:
        return unavailable
    if (value.get("token_proxy") != "utf8-bytes-ceil-div4@1"
            or type(value.get("selected_mode")) is not str
            or value["selected_mode"] not in {"legacy", "common-bindings@1"}):
        return unavailable
    result = {**unavailable, "status": "reported", "schema": schema,
        "selected_mode": value["selected_mode"], "token_proxy": value["token_proxy"],
        "selection_basis": "byte_proxy", "measurement_scope": "complete-rendered-prompt-text@1"}
    for field in ("view_receipt_sha256", "native_complete_sha256", "candidate_complete_sha256", "selected_complete_sha256"):
        digest = value.get(field)
        if type(digest) is not str or re.fullmatch("[0-9a-f]{64}", digest) is None:
            return unavailable
        result[field] = digest
    for prefix in ("native", "candidate", "selected"):
        size, proxy = value.get(prefix + "_complete_bytes"), value.get(prefix + "_complete_proxy_tokens")
        if (type(size) is not int or not 1 <= size <= 2_000_000
                or type(proxy) is not int or proxy != (size + 3) // 4):
            return unavailable
        result[prefix + "_complete_bytes"], result[prefix + "_complete_proxy_tokens"] = size, proxy
    chosen = "candidate" if result["selected_mode"] == "common-bindings@1" else "native"
    if any(result["selected_complete_" + field] != result[chosen + "_complete_" + field]
           for field in ("sha256", "bytes", "proxy_tokens")):
        return unavailable
    reason = value.get("fallback_reason")
    allowed_reasons = {None, "complete_input_not_smaller_under_bytes_and_proxy"}
    if schema.endswith("@2"):
        allowed_reasons = {None, "complete_input_not_smaller_under_bytes_and_token_counter",
            "complete_input_token_counter_invalid", "complete_input_token_counter_failed"}
    if type(reason) not in (str, type(None)) or reason not in allowed_reasons:
        return unavailable
    if ((chosen == "candidate") != (reason is None)
            or (chosen == "candidate" and result["candidate_complete_bytes"] >= result["native_complete_bytes"])):
        return unavailable
    result["fallback_reason"] = reason
    if schema.endswith("@1"):
        return result
    identity = value.get("token_counter_identity")
    status = value.get("token_counter_status")
    if (type(status) is not str or status not in {"counted", "failed", "invalid"}
            or value.get("selection_basis") != "token_counter"):
        return unavailable
    result.update(selection_basis="token_counter", token_counter_status=status,
        token_counter_identity=None, token_counter_identity_sha256=None,
        native_complete_tokens=None, candidate_complete_tokens=None, selected_complete_tokens=None)
    if status == "invalid":
        return result if (identity is None and value.get("token_counter_identity_sha256") is None
            and reason == "complete_input_token_counter_invalid"
            and all(value.get(field) is None for field in (
                "native_complete_tokens", "candidate_complete_tokens", "selected_complete_tokens"))) else unavailable
    if (type(identity) is not dict or identity.get("schema") != "supervisor-semantic-metadata-token-counter@1"
            or identity.get("measurement_scope") != result["measurement_scope"]
            or type(identity.get("qualification")) is not str
            or identity["qualification"] not in {"provider_tokenizer", "estimate"}
            or identity.get("identity_caller_supplied") is not True
            or identity.get("provider_usage_verified") is not False):
        return unavailable
    safe_identity = {"schema": identity["schema"], "qualification": identity["qualification"],
        "measurement_scope": result["measurement_scope"], "identity_caller_supplied": True,
        "provider_usage_verified": False}
    for field in ("provider", "model", "tokenizer", "tokenizer_revision"):
        identifier = identity.get(field)
        if type(identifier) is not str or re.fullmatch("[A-Za-z0-9][A-Za-z0-9._:/@+\\-]{0,255}", identifier) is None:
            return unavailable
        safe_identity[field] = identifier
    digest = value.get("token_counter_identity_sha256")
    canonical = json.dumps(safe_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if digest != hashlib.sha256(canonical.encode()).hexdigest():
        return unavailable
    result.update(token_counter_identity=safe_identity, token_counter_identity_sha256=digest)
    if status == "failed":
        return result if reason == "complete_input_token_counter_failed" and all(value.get(field) is None for field in (
            "native_complete_tokens", "candidate_complete_tokens", "selected_complete_tokens")) else unavailable
    for prefix in ("native", "candidate", "selected"):
        tokens = value.get(prefix + "_complete_tokens")
        if type(tokens) is not int or not 1 <= tokens <= 2**63 - 1:
            return unavailable
        result[prefix + "_complete_tokens"] = tokens
    if result["selected_complete_tokens"] != result[chosen + "_complete_tokens"]:
        return unavailable
    if reason not in {None, "complete_input_not_smaller_under_bytes_and_token_counter"}:
        return unavailable
    if chosen == "candidate" and result["candidate_complete_tokens"] >= result["native_complete_tokens"]:
        return unavailable
    return result


def _input_observation(raw):
    """Project reported wire measurements without exporting prompt contents."""
    translation = _mapping(raw.get("semantic_translation"))
    residual = _mapping(raw.get("doctor_residual_context"))
    result = {
        **{key: _count(raw.get(key)) for key in (
            "native_prompt_bytes", "router_prompt_bytes", "model_prompt_bytes", "workspace_advisory_bytes")},
        **{key: _text(raw.get(key)) for key in (
            "native_prompt_sha256", "router_prompt_sha256", "model_prompt_sha256", "workspace_advisory_sha256")},
        "semantic_translation": {
            **{key: _text(translation.get(key)) for key in ("schema", "translation_cid")},
            **{key: _count(translation.get(key)) for key in (
                "native_prompt_bytes", "provider_prompt_bytes", "identifier_mappings", "identifier_occurrences")},
            "freshness_checked": _boolean(translation.get("freshness_checked")),
        } if translation else None,
        "doctor_residual_context": {
            **{key: _text(residual.get(key)) for key in (
                "schema", "context_cid", "native_capsule_id", "advisory_sha256")},
            **{key: _count(residual.get(key)) for key in ("advisory_bytes", "extra_provider_calls")},
            **{key: _boolean(residual.get(key)) for key in (
                "source_freshness_verified", "candidate_only", "derived_runtime_admitted")},
        } if residual else None,
        "measurements_are_token_savings": False,
        "input_reconstruction_verified_by_receipt_presence": False,
    }
    if "semantic_metadata_view" in raw:
        result["semantic_metadata_view"] = _semantic_metadata_observation(raw["semantic_metadata_view"])
    return result


def _audit_observation(report):
    audit = _mapping(report.get("context_input_audit"))
    return {
        **{key: _text(audit.get(key)) for key in ("status", "reason", "error_type")},
        **{key: _boolean(audit.get(key)) for key in (
            "any_native_input_verified", "any_model_input_verified", "all_observed_coding_inputs_verified")},
        **{key: _count(audit.get(key)) for key in ("coding_receipts_seen", "usable_coding_receipts")},
        "seconds": _number(audit.get("seconds")),
        "invocations": [{
            **{key: _text(item.get(key)) for key in ("invocation_id", "status", "reason")},
            **{key: _boolean(item.get(key)) for key in ("native_input_verified", "model_input_verified")},
        } for item in audit.get("invocations", []) if isinstance(item, dict)][:16]
            if isinstance(audit.get("invocations"), list) else [],
        "task_correctness_established": False,
    } if audit else None


def _declared_output_observation(report):
    """Retain closed byte facts separately from native settlement and reward."""
    raw = _mapping(report.get("declared_output_observation"))
    result = {"schema": "terminal-declared-output-observation@1", "status": "unavailable",
        "reason": "not_recorded", "outputs": [], "observation_only": True,
        "completion_authority": False, "settlement_authority": False,
        "task_correctness_established": False, "patch_application_established": False,
        "writer_identity_established": False}
    if not raw:
        return result
    result["reason"] = "invalid_observation"
    if (raw.get("schema") != result["schema"] or type(raw.get("status")) is not str
            or raw["status"] not in {"observed", "partial", "unavailable"}):
        return result
    def digest(value, length=64):
        return value if type(value) is str and re.fullmatch("[0-9a-f]{%d}" % length, value) else None
    def file_fact(value):
        value = _mapping(value)
        unavailable = dict(status="unavailable", exists=None, size_bytes=None, sha256=None, equals_admitted_baseline=None)
        if value.get("status") != "observed" or type(value.get("exists")) is not bool:
            return unavailable
        present = value["exists"]
        size, hashed = _count(value.get("size_bytes")), digest(value.get("sha256"))
        if (present and (size is None or size > 1_000_000 or hashed is None)
                or not present and (value.get("size_bytes") is not None or value.get("sha256") is not None)):
            return unavailable
        return dict(status="observed", exists=present, size_bytes=size, sha256=hashed,
            equals_admitted_baseline=_boolean(value.get("equals_admitted_baseline")))
    rows = raw.get("outputs")
    if type(rows) is not list or len(rows) > 32:
        return result
    paths = set()
    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import _path
        for row in rows:
            if type(row) is not dict or type(row.get("path")) is not str or len(row["path"].encode()) > 1024:
                raise ValueError("invalid output row")
            path = _path(row["path"])
            if path == "." or path in paths or row.get("effect") not in {"create", "modify", "delete"}:
                raise ValueError("invalid output scope")
            paths.add(path)
            result["outputs"].append(dict(path=path, effect=row["effect"], baseline_sha256=digest(row.get("baseline_sha256")),
                candidate=file_fact(row.get("candidate")), canonical=file_fact(row.get("canonical"))))
    except (TypeError, ValueError):
        result["outputs"] = []
        return result
    result.update(status=raw["status"], reason="reported_declared_byte_observation",
        **{key: digest(raw.get(key)) for key in ("task_cid_sha256", "attempt_id_sha256", "invocation_id_sha256",
            "allocation_path_sha256", "owner_events_sha256")}, baseline_commit=digest(raw.get("baseline_commit"), 40),
        portal_attempt=_count(raw.get("portal_attempt")), original_allocation_inode_verified=False)
    # Malformed individual facts cannot retain a blanket complete status.
    if result["status"] == "observed" and (not rows or any(row[label]["status"] != "observed"
            for row in result["outputs"] for label in ("candidate", "canonical"))):
        result["status"] = "partial" if rows else "unavailable"
    return result


def _doctor_observation(report):
    doctor = _mapping(report.get("doctor_dispatch"))
    invocations = report.get("doctor_invocations")
    invocations = invocations if isinstance(invocations, list) else []
    observed = [item for item in invocations if isinstance(item, dict)
        and item.get("schema") in {"native-doctor-candidate-materialization@1", "native-doctor-contract-candidate-materialization@1"}
        and item.get("task_cid") == doctor.get("task_cid")]
    return {
        "route": _text(report.get("implementation_route")),
        "status": _text(doctor.get("status")),
        **{key: _count(doctor.get(key)) for key in (
            "provider_calls", "residual_successors", "residual_work_proposals")},
        "residual_context_prepared": bool(doctor.get("residual_context")) if doctor else None,
        "observed_worker_receipts": len(observed) if isinstance(report.get("doctor_invocations"), list) else None,
        "candidate_materializations_observed": sum(item.get("status") == "candidate_materialized" for item in observed)
            if isinstance(report.get("doctor_invocations"), list) else None,
        "selection_is_publication_or_completion_evidence": False,
        "local_contract_proof_status": _text(_mapping(_mapping(doctor.get("contract_workflow")).get("proof")).get("status")),
        "whole_program_proved": False,
    }


def _code_learning_observation(report):
    initial = _mapping(report.get("initial_context"))
    learner = _mapping(initial.get("codebase_autoencoder"))
    if not learner:
        return None
    metrics = _mapping(learner.get("metrics"))
    catalog = _mapping(initial.get("codebase_autoencoder_catalog"))
    transfer = _mapping(learner.get("weight_transfer"))
    security = _mapping(metrics.get("security_candidate_training"))
    return {
        **{key: _text(learner.get(key)) for key in (
            "schema", "domain", "checkpoint_sha256", "receipt_sha256")},
        **{key: _count(learner.get(key)) for key in ("sample_count", "epochs_completed")},
        "training_elapsed_seconds": _number(learner.get("training_elapsed_seconds")),
        "native_kernel_calls": _count(metrics.get("native_kernel_calls")),
        "reconstruction_loss_before": _number(metrics.get("before_reconstruction_loss")),
        "reconstruction_loss_after": _number(metrics.get("after_reconstruction_loss")),
        "holdout_evaluated": _boolean(metrics.get("holdout_evaluated")),
        "weight_transfer": {
            **{key: _text(transfer.get(key)) for key in (
                "source_checkpoint_sha256", "initializer_sha256", "runtime_validation_scope")},
            "transferred_row_count": _count(transfer.get("transferred_row_count")),
            "random_initialization": _boolean(_mapping(metrics.get("weight_transfer")).get("random_initialization")),
            "legal_state_mutation_authorized": False,
        } if transfer else None,
        "canonical_cve_training": {
            "manifest_sha256": _text(_mapping(learner.get("canonical_cve_training")).get("manifest_sha256")),
            "sample_count": _count(security.get("sample_count")),
            "training_bce_before": _number(_mapping(security.get("before")).get("training_bce")),
            "training_bce_after": _number(_mapping(security.get("after")).get("training_bce")),
            "holdout_evaluated": _boolean(security.get("holdout_evaluated")),
            "classification_is_formal_translation": False,
        } if security else None,
        "task_candidate_rows": (len(_mapping(learner.get("security_candidate_nominations")).get("rows", []))
            if learner.get("security_candidate_nominations") else None),
        "catalog_status": _text(catalog.get("status")),
        "new_training_steps_after_admission": _count(_mapping(report.get("context")).get("new_autoencoder_training_steps")),
        "post_publication_status": _text(_mapping(report.get("post_publication_autoencoder")).get("status")),
        "post_publication_freshness_checked": _boolean(_mapping(report.get("post_publication_autoencoder")).get("freshness_checked")),
        "formal_translation_authority": False,
        "reconstruction_improvement_is_security_accuracy": False,
    }


def _proof_index_observation(report):
    refresh = _mapping(report.get("post_publication_proof_index"))
    if not refresh:
        return None
    return {"status": _text(refresh.get("status")),
        "active_receipts": len(refresh["active_receipt_ids"]) if isinstance(refresh.get("active_receipt_ids"), list) else None,
        "invalidated_receipts": len(refresh["invalidated_receipt_ids"]) if isinstance(refresh.get("invalidated_receipt_ids"), list) else None,
        "new_proof_authority": False}


def _refresh_observation(report):
    refresh = _mapping(report.get("post_publication_context"))
    accounting = _mapping(refresh.get("embedding_accounting"))
    complete = _boolean(accounting.get("all_refreshes_receipted"))
    return {
        **{key: _text(refresh.get(key)) for key in ("status", "reason", "error_type")},
        **{key: _number(refresh.get(key)) for key in ("budget_seconds", "refresh_seconds")},
        "embedding_calls": _count(refresh.get("embedding_calls")) if complete is True else None,
        "embedding_accounting": {
            "all_refreshes_receipted": complete,
            "totals": {key: _count(_mapping(accounting.get("totals")).get(key))
                if complete is True else None for key in EMBEDDING_COUNTERS},
            "known_subtotals": {key: _count(_mapping(accounting.get("known_subtotals")).get(key))
                for key in EMBEDDING_COUNTERS},
        },
        "task_completion_is_context_refresh_evidence": False,
    } if refresh else None


def _sum(values):
    values = list(values)
    return sum(values) if values and all(value is not None for value in values) else None


def _subtotal(values):
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _grok_progress(raw):
    value = raw.get("native_grok_progress")
    if value is None:
        return None
    from ipfs_accelerate_py.cli_runtime.grok_stream_capture import validate_grok_stream_progress
    try:
        return validate_grok_stream_progress(value)
    except (TypeError, ValueError):
        return None


def _uniform_boolean(values):
    values = list(values)
    return values[0] if values and type(values[0]) is bool and all(value is values[0] for value in values) else None


def _invocations(report):
    found = {}
    rows = report.get("provider_invocations", [])
    if not isinstance(rows, list):
        raise ValueError("provider invocation receipts must be a list")
    for raw in rows:
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("invocation_id"), str)
            or not raw["invocation_id"]
        ):
            raise ValueError("provider invocation identity is required")
        identity = raw["invocation_id"]
        if identity in found:
            if found[identity][0] != raw:
                raise ValueError("conflicting provider invocation receipts")
            continue
        native = _mapping(raw.get("native_rollout_usage"))
        observed = _tokens(native.get("usage"))
        complete, scope, cache_included = native_usage_accounting.native_usage_contract(raw, native)
        observed_cost = _number(native.get("dollar_cost"))
        found[identity] = (
            raw,
            {
                "invocation_id": _text(identity),
                "phase": _text(raw.get("phase")),
                "provider": _text(raw.get("provider")),
                "model": _text(raw.get("model")),
                "status": _text(raw.get("status")),
                "error_type": _text(raw.get("error_type")),
                "timeout_seconds": _number(raw.get("timeout_seconds")),
                "seconds": _number(raw.get("seconds")),
                "tokens": {key: value if complete is True else None for key, value in observed.items()},
                "known_token_subtotals": observed,
                "native_task_complete_observed": _boolean(native.get("task_complete_observed")),
                "native_usage_complete_observed": complete,
                "token_count_scope": scope,
                "cache_included_in_input": cache_included,
                "observed_counts_are_lower_bounds": scope == "native_session_cumulative",
                "billing_total_verified": False,
                "reported_cost_usd": observed_cost if complete is True else None,
                "known_reported_cost_subtotal_usd": observed_cost,
                "input_observation": _input_observation(raw),
                "native_grok_progress": _grok_progress(raw),
            },
        )
    return [item[1] for item in found.values()]


def _trial(receipt, raw, *, source):
    baseline = receipt["schema"] == BASELINE
    arm = "native-codex" if baseline else receipt["arm"]
    durations = raw.get("seconds" if baseline else "durations_seconds") or {}
    report = raw.get("supervisor") if not baseline else None
    report = report if isinstance(report, dict) else {}
    context = _mapping(report.get("context"))
    doctor = _mapping(context.get("doctor_repair"))
    initial = _mapping(report.get("initial_context"))
    planning_index = _mapping(_mapping(report.get("planning")).get("initial_indexed_context"))
    phases = _mapping(report.get("phases"))
    exception = _text(raw.get("exception_type"))
    rewards = raw.get("reward")
    reward = _number(rewards.get("reward")) if isinstance(rewards, dict) else None
    completed = report.get("task_completed") if type(report.get("task_completed")) is bool else None
    if reward == 1:
        outcome = "official_verifier_passed"
    elif reward is not None:
        outcome = "official_verifier_not_passed"
    elif exception and not report and durations.get("agent_execution") is None:
        outcome = "setup_or_environment_aborted"
    elif exception or report.get("error"):
        outcome = "failed_without_verifier_reward"
    else:
        outcome = "outcome_unknown"
    invocations = _invocations(report) if not baseline else []
    unreceipted = bool(report.get("unreceipted_provider_attempt"))
    if baseline:
        native = _mapping(raw.get("raw_usage"))
        completion = _baseline_completion(native)
        subtotal = _tokens(native.get("usage"))
        tokens = {key: value if completion is True else None for key, value in subtotal.items()}
        sessions = native.get("sessions")
        calls = len(sessions) if isinstance(sessions, list) and sessions else None
        cost_subtotal = _number(native.get("reported_cost_usd"))
        cost = cost_subtotal if completion is True else None
        cache_included, lower_bounds = True, True
    else:
        completion = None if unreceipted else _completion(
            row["native_usage_complete_observed"] for row in invocations)
        subtotal = {key: _subtotal(row["known_token_subtotals"][key] for row in invocations) for key in TOKENS}
        tokens = {
            key: None if unreceipted else _sum(row["tokens"][key] for row in invocations)
            for key in TOKENS
        }
        calls = len(invocations) if invocations and not unreceipted else None
        cost = None if unreceipted else _sum(row["reported_cost_usd"] for row in invocations)
        cost_subtotal = _subtotal(row["known_reported_cost_subtotal_usd"] for row in invocations)
        cache_included = _uniform_boolean(row["cache_included_in_input"] for row in invocations)
        lower_bounds = bool(invocations) and all(row["observed_counts_are_lower_bounds"] for row in invocations)
    return {
        "arm": arm,
        "label": LABELS[arm],
        "trial": _text(raw.get("trial")),
        "source_receipts": [source],
        "task": _text(receipt.get("task")),
        "model": _text(receipt.get("model")),
        "reasoning_effort": _text(receipt.get("reasoning_effort")),
        "cli_version": _text(receipt.get("cli_version")),
        "runtime_archive_sha256": _archive_digest(raw) if not baseline else None,
        "original_task_inputs_unchanged": receipt.get("original_task_inputs_unchanged") is True,
        "native_result_sha256": _text(
            raw.get("result_sha256" if baseline else "native_result_sha256")
        ),
        "official_reward": reward,
        "native_task_completed": completed,
        "native_completion_applicable": not baseline,
        "outcome": outcome,
        "exception_type": exception,
        "supervisor_error_type": _text((report.get("error") or {}).get("type")),
        "durations_seconds": {key: _number(durations.get(key)) for key in PHASES},
        "native_total_seconds": _number(raw.get("total_seconds")),
        "supervisor_seconds": _number(report.get("seconds")),
        "preparation_seconds": _number(phases.get("prepare_seconds")),
        "context_seconds": _number(phases.get("context_seconds")),
        "supervisor_phases_seconds": {key: _number(phases.get(key)) for key in SUPERVISOR_PHASES}
            if not baseline else None,
        "initial_indexed_context_observation": {
            "prepared": bool(initial),
            "learned_embeddings": _boolean(initial.get("learned_embeddings")),
            **{key: _count(initial.get(key)) for key in (
                "indexed_symbols", "full_capsules", "world_task_count", "provider_calls")},
            "canonical_tasks_created": _boolean(initial.get("canonical_tasks_created")),
            "supplied_to_planning_router": _boolean(planning_index.get("supplied_to_router")),
            "planning_model_request_sha256": _text(_mapping(report.get("planning")).get("model_request_sha256")),
            "summary_sha256": _text(planning_index.get("summary_sha256")),
            "preparation_is_provider_dispatch_evidence": False,
        } if not baseline else None,
        "indexed_context_observation": {
            "prepared": bool(context),
            "learned_embeddings": _boolean(context.get("learned_embeddings")),
            **{key: _number(context.get(key)) for key in (
                "indexed_symbols", "native_fact_rows_replayed", "full_capsules",
                "worker_capsules", "worker_semantic_bytes",
            )},
            "doctor_repair_status": _text(doctor.get("status")),
            "initial_indexes_reused": _boolean(context.get("initial_indexes_reused")),
            "new_embedding_calls": _count(context.get("new_embedding_calls")),
            "context_preparation_is_dispatch_evidence": False,
        } if not baseline else None,
        "doctor_observation": _doctor_observation(report) if not baseline else None,
        "code_learning_observation": _code_learning_observation(report) if not baseline else None,
        "post_publication_proof_index": _proof_index_observation(report) if not baseline else None,
        "context_input_audit": _audit_observation(report) if not baseline else None,
        "declared_output_observation": _declared_output_observation(report) if not baseline else None,
        "post_publication_context": _refresh_observation(report) if not baseline else None,
        "provider_calls": calls,
        "observed_provider_invocations": len(invocations) if not baseline else calls,
        "unreceipted_provider_attempt": unreceipted,
        "native_usage_complete_observed": completion,
        "cache_included_in_input": cache_included,
        "observed_counts_are_lower_bounds": lower_bounds,
        "billing_total_verified": False,
        "tokens": tokens,
        "known_token_subtotals": subtotal,
        "reported_cost_usd": cost,
        "known_reported_cost_subtotal_usd": cost_subtotal,
        "invocations": invocations,
        "retained_trial": True,
    }


def _aggregate(rows):
    return {
        "trials": len(rows),
        "observed_official_passes": sum(row["official_reward"] == 1 for row in rows),
        "unknown_reward_trials": sum(row["official_reward"] is None for row in rows),
        "provider_calls": _sum(row["provider_calls"] for row in rows),
        "tokens": {key: _sum(row["tokens"][key] for row in rows) for key in TOKENS},
        "known_token_subtotals": {
            key: _subtotal(row["known_token_subtotals"][key] for row in rows) for key in TOKENS
        },
        "unknown_token_trials": {
            key: sum(row["tokens"][key] is None for row in rows) for key in TOKENS
        },
        "reported_cost_usd": _sum(row["reported_cost_usd"] for row in rows),
        "known_reported_cost_subtotal_usd": _subtotal(row["known_reported_cost_subtotal_usd"] for row in rows),
    }


def collect(*, baseline: Path, supervisors: list[Path]) -> dict:
    sources, rows, seen = [], [], {}
    profile = None
    baseline_controls = None
    baseline_integrity = None
    for index, path in enumerate([baseline, *supervisors]):
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        if len(raw) > 64 * 1024 * 1024:
            raise ValueError("receipt exceeds comparison input bound")
        receipt = json.loads(raw)
        expected = BASELINE if index == 0 else SUPERVISOR
        if receipt.get("schema") != expected:
            raise ValueError("comparison requires native baseline and supervisor receipts")
        if not index:
            baseline_controls = receipt.get("comparison_controls")
            baseline_integrity = _boolean(receipt.get("original_task_inputs_unchanged"))
            profile = tuple(
                receipt.get(key) for key in ("task", "model", "reasoning_effort", "cli_version")
            )
        elif receipt.get("arm") not in ("full", "no-index"):
            raise ValueError("unknown supervisor arm")
        trials = receipt.get("trials")
        if not isinstance(trials, list) or receipt.get("trial_count") != len(trials):
            raise ValueError("receipt trial inventory is inconsistent")
        source = {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "trial_count": len(trials),
            "invocation_seconds": _number(receipt.get("invocation_seconds")),
        }
        sources.append(source)
        for trial in trials:
            if (
                not isinstance(trial, dict)
                or not isinstance(trial.get("trial"), str)
                or not trial["trial"]
            ):
                raise ValueError("retained native trial identity required")
            row = _trial(receipt, trial, source=source)
            row["reported_identity_matches"] = (
                tuple(
                    receipt.get(key) for key in ("task", "model", "reasoning_effort", "cli_version")
                )
                == profile
            )
            row["declared_controls_comparison"] = compare_controls(
                baseline_controls, receipt.get("comparison_controls"))
            row["comparison_profile_matches"] = row["declared_controls_comparison"]["matches"]
            declared_identity = _mapping(_mapping(_mapping(receipt.get("comparison_controls")).get("declared")).get("identity"))
            row["declaration_identity_matches_report"] = (
                all(declared_identity.get(key) == receipt.get(key) for key in
                    ("task", "model", "reasoning_effort", "cli_version")) if declared_identity else None)
            if row["declaration_identity_matches_report"] is False:
                row["comparison_profile_matches"] = False
            if not row["reported_identity_matches"] or receipt.get("original_task_inputs_unchanged") is False:
                row["comparison_profile_matches"] = False
            if baseline_integrity is False:
                row["comparison_profile_matches"] = False
            elif row["comparison_profile_matches"] is True and (
                baseline_integrity is not True or receipt.get("original_task_inputs_unchanged") is not True
            ):
                row["comparison_profile_matches"] = None
            identity = (row["arm"], row["task"], row["trial"])
            previous = seen.get(identity)
            if previous:
                if {key: value for key, value in previous.items() if key != "source_receipts"} != {
                    key: value for key, value in row.items() if key != "source_receipts"
                }:
                    raise ValueError("conflicting retained trial identity")
                previous["source_receipts"].append(source)
            else:
                seen[identity] = row
                rows.append(row)
    unsuccessful = [
        row for row in rows if row["outcome"] not in ("official_verifier_passed", "outcome_unknown")
    ]
    return {
        "schema": "terminal-supervision-comparison@1",
        "comparison_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_usage_accounting_source_sha256": hashlib.sha256(Path(native_usage_accounting.__file__).read_bytes()).hexdigest(),
        "sources": sources,
        "rows": rows,
        "retained_trial_count": len(rows),
        "receipts_without_trials": [row for row in sources if not row["trial_count"]],
        "arm_totals": {
            arm: _aggregate([row for row in rows if row["arm"] == arm])
            for arm in LABELS
            if any(row["arm"] == arm for row in rows)
        },
        "unsuccessful_attempt_costs": {
            arm: _aggregate([row for row in unsuccessful if row["arm"] == arm])
            for arm in LABELS
            if any(row["arm"] == arm for row in unsuccessful)
        },
        "benchmark_advantage_claimed": False,
        "all_declared_controls_match": bool(rows) and all(
            row["comparison_profile_matches"] is True for row in rows),
        "runtime_control_enforcement_verified": False,
        "causal_claim": False,
        "unknown_fields_are_zero": False,
        "cached_tokens_are_already_in_input": _uniform_boolean(row["cache_included_in_input"] for row in rows),
        "provider_call_count_basis": "native CLI sessions for baseline; router invocations for supervisor; not internal model turns",
        "dollar_cost_basis": "reported provider cost only; Harbor estimates excluded",
        "notes": [
            "Every supplied retained trial is shown, including setup aborts and unsuccessful planning or coding.",
            "No-index runs the native supervisor without an indexed context bundle; planning strategies may differ between workflows.",
            "Native task completion and the independent original verifier reward are separate outcomes.",
            "Known subtotals preserve observed failed-call costs when complete usage is unavailable.",
            "Codex cumulative counters require native session completion; Grok final-envelope counters additionally require explicit usage completeness and coherent native categories. Unknown receipts retain observations only.",
            "Grok final-envelope observations are not asserted to be cumulative lower bounds or complete whole-session billing; omitted completeness flags stay unknown.",
            "Runtime archive hashes are reported from retained agent metadata; the comparison source hash identifies this reporting implementation separately.",
            "Initial cold indexing, planning, Doctor and post-publication refresh are inside supervisor agent time; reported phases are observations, not added to total time or nested helper durations.",
            "Prepared context, reported wire measurements, and independently reconstructed model inputs are separate observations; bytes do not establish token savings.",
            "Native counters are counted once per invocation. Codex input includes cache; Grok input excludes cache. Mixed-provider cache semantics are unknown at the comparison level.",
            "A single-task pilot does not establish efficiency, parallelism, or benchmark superiority.",
            "Matching declarations cover task bytes, timeouts, retries, resources and concurrency; they do not establish runtime enforcement. Legacy preparations without frozen controls remain unknown.",
        ],
    }


def _cell(value):
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.3f}"
    return html.escape(str(value)).replace("|", "\\|").replace("\n", " ")


def _token_cell(row, key):
    value = row["tokens"][key]
    if value is not None:
        return value
    observed = row["known_token_subtotals"][key]
    prefix = "≥ " if row.get("observed_counts_are_lower_bounds") is True else ""
    return f"{prefix}{observed} (observed)" if observed is not None else None


def markdown(comparison: dict) -> str:
    lines = [
        "Retained pilot trials. These observations do not establish a benchmark advantage.",
        "",
        "Token cells marked ≥ are Codex cumulative lower bounds; complete totals remain unknown. Other cells marked observed do not assert a lower bound. Grok additionally requires explicit usage completeness. Verifier reward and supervisor task completion do not establish usage completeness or billing totals.",
        "",
        "| Arm | Trial | Reward | Native completion | Env setup s | Agent setup s | Agent s | Verifier s | Input | Cached | Output | Total | Outcome |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in comparison["rows"]:
        values = [
            row["label"],
            row["trial"],
            row["official_reward"],
            row["native_task_completed"] if row["native_completion_applicable"] else "N/A",
            *(row["durations_seconds"][key] for key in PHASES),
            *(_token_cell(row, key) for key in TOKENS),
            row["outcome"],
        ]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    lines += ["", "Declared controls are checked separately from reported model identity. A match is a configuration comparison, not runtime enforcement or campaign qualification.", "",
              "| Trial | Reported identity matches | Declared controls match | Differences |",
              "|---|---|---|---|"]
    for row in comparison["rows"]:
        lines.append("| " + " | ".join(_cell(value) for value in (
            row["trial"], row["reported_identity_matches"], row["comparison_profile_matches"],
            ", ".join(row["declared_controls_comparison"]["differences"]))) + " |")
    lines += [
        "",
        "Unsuccessful attempt costs are retained separately. Values below are observed subtotals; unknown calls or counters can make the actual total larger.",
        "",
        "| Arm | Unsuccessful trials | Known input | Known cached | Known output | Known total | Trials with unknown totals |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, totals in comparison["unsuccessful_attempt_costs"].items():
        values = [
            LABELS[arm],
            totals["trials"],
            *(totals["known_token_subtotals"][key] for key in TOKENS),
            totals["unknown_token_trials"]["total_tokens"],
        ]
        lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    lines += [
        "",
        "Codex input includes cached tokens; Grok input excludes them. Each row retains native categories. Missing values remain unknown. Provider-reported dollar costs are unavailable when shown as null in JSON; estimated Harbor costs are excluded.",
        "",
        "No-index runs the native supervisor without an indexed context bundle; planning strategies may differ between workflows. Setup, agent execution, and verification remain separate timings.",
    ]
    supervised = [row for row in comparison["rows"] if row["native_completion_applicable"]]
    calls = [(row, call) for row in supervised for call in row["invocations"]]
    if calls:
        lines += ["", "Provider invocation observations retain failure and timeout information independently of the eventual task outcome.", "",
                  "| Trial | Invocation | Phase | Status | Error | Timeout s | Native session complete | Usage complete | Count scope | Input | Cached | Output | Total |",
                  "|---|---|---|---|---|---:|---|---|---|---:|---:|---:|---:|"]
        for row, call in calls:
            values = [row["trial"], call["invocation_id"], call["phase"], call["status"],
                call["error_type"], call["timeout_seconds"], call["native_task_complete_observed"],
                call["native_usage_complete_observed"], call["token_count_scope"],
                *(_token_cell(call, key) for key in TOKENS)]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    archives = [row for row in supervised if row["runtime_archive_sha256"] is not None]
    if archives:
        lines += ["", "Runtime archive provenance is projected from retained agent metadata. The comparison implementation has its own source SHA256 in comparison.json; it may postdate these immutable runtimes.", "",
                  "| Trial | Runtime archive SHA256 |", "|---|---|"]
        for row in archives:
            lines.append("| " + " | ".join(_cell(row[key]) for key in ("trial", "runtime_archive_sha256")) + " |")
    if supervised:
        lines += ["", "Supervisor phase observations are already inside agent time. Cold initial indexing is included; these values are not added to the total or to nested helper timings. Missing phases remain unknown.", "",
                  "| Trial | Prepare s | Cold initial context s | Planning s | Admitted context s | Doctor s | Post-publication refresh s |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in supervised:
            values = [row["trial"], *(row["supervisor_phases_seconds"][key] for key in SUPERVISOR_PHASES)]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    indexed = [row for row in comparison["rows"]
               if (row.get("indexed_context_observation") or {}).get("prepared")]
    if indexed:
        lines += ["", "Indexed preparation is charged to agent time. These counts describe prepared context; they do not by themselves prove worker consumption or repair execution.", "",
                  "| Trial | Initial indexes reused | Vector rows replayed | Full capsules | Worker capsules | Worker semantic bytes | Doctor eligibility status |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for row in indexed:
            context = row["indexed_context_observation"]
            values = [row["trial"], context["initial_indexes_reused"], *(context[key] for key in (
                "native_fact_rows_replayed", "full_capsules", "worker_capsules",
                "worker_semantic_bytes", "doctor_repair_status"))]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    if supervised:
        lines += ["", "Doctor selection, worker materialization and context refresh remain separate from native completion and verifier reward. Embedding subtotals retain measured failed work when complete accounting is unavailable.", "",
                  "| Trial | Implementation route | Doctor status | Worker materializations observed | Refresh status | Local embedding calls | Known local call subtotal |",
                  "|---|---|---|---:|---|---:|---:|"]
        for row in supervised:
            doctor = row["doctor_observation"]
            refresh = row["post_publication_context"] or {}
            accounting = refresh.get("embedding_accounting", {})
            values = [row["trial"], doctor["route"], doctor["status"], doctor["candidate_materializations_observed"],
                refresh.get("status"), accounting.get("totals", {}).get("local_embedding_calls"),
                accounting.get("known_subtotals", {}).get("local_embedding_calls")]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
        lines += ["", "Model-input audit reconstructs retained context and exact task-prompt bytes. It does not cover provider system context or establish task correctness.", "",
                  "| Trial | Audit status | Any native input verified | Any model input verified | All observed coding inputs verified |",
                  "|---|---|---|---|---|"]
        for row in supervised:
            audit = row["context_input_audit"] or {}
            values = [row["trial"], *(audit.get(key) for key in (
                "status", "any_native_input_verified", "any_model_input_verified", "all_observed_coding_inputs_verified"))]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    inputs = [(row, call) for row in supervised for call in row["invocations"]
              if any(call["input_observation"][key] is not None for key in (
                  "native_prompt_bytes", "router_prompt_bytes", "model_prompt_bytes"))]
    if inputs:
        lines += ["", "Reported task-prompt sizes are bytes, not token savings. Router bytes include semantic translation and any Doctor residual advisory; model bytes also include the workspace instruction.", "",
                  "| Trial | Invocation | Phase | Native bytes | Router bytes | Model bytes | Identifier mappings | Residual advisory bytes |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
        for row, call in inputs:
            observed = call["input_observation"]
            values = [row["trial"], call["invocation_id"], call["phase"],
                *(observed[key] for key in ("native_prompt_bytes", "router_prompt_bytes", "model_prompt_bytes")),
                (observed["semantic_translation"] or {}).get("identifier_mappings"),
                (observed["doctor_residual_context"] or {}).get("advisory_bytes")]
            lines.append("| " + " | ".join(_cell(value) for value in values) + " |")
    if comparison["receipts_without_trials"]:
        lines += [
            "",
            f"{len(comparison['receipts_without_trials'])} supplied receipts contained no native trial record; they remain listed in comparison.json.",
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh exact comparison output directory required")
    comparison = collect(baseline=args.baseline, supervisors=args.supervisor)
    output.mkdir(parents=True)
    (output / "comparison.json").write_text(json.dumps(comparison, sort_keys=True, indent=2) + "\n")
    (output / "comparison.md").write_text(markdown(comparison))
    print(
        json.dumps(
            {
                "retained_trials": comparison["retained_trial_count"],
                "benchmark_advantage_claimed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
