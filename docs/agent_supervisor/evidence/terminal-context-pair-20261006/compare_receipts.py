"""Compare a fresh @1/@2 pair using bounded, public metadata only.

Keep both planning and coding sessions. Never read transcripts, logs, model
thoughts, solutions, source bodies or hidden evaluator contents. Interrupted
and failed attempts remain visible with unknown complete usage and observed
subtotals. Cached input already belongs to the input total.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys

MAX_BYTES = 64 * 1024 * 1024
TRANSPORTS = ("supervisor-semantic-router-input@1", "supervisor-semantic-router-input@2")
TOKENS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load(path, bindings):
    path = Path(path).absolute()
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or path.resolve(strict=True) != path
            or info.st_size > MAX_BYTES):
        raise ValueError("bounded canonical metadata file required")
    raw = path.read_bytes()
    after = path.stat()
    if (len(raw) != info.st_size or after.st_mtime_ns != info.st_mtime_ns
            or after.st_ino != info.st_ino):
        raise ValueError("metadata changed during read")
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate metadata key")
            result[key] = item
        return result
    result = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite metadata")))
    if type(result) is not dict:
        raise ValueError("metadata object required")
    binding = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    bindings.append(binding)
    return result, binding


def mapping(value):
    return value if type(value) is dict else {}


def count(value):
    return value if type(value) is int and value >= 0 else None


def sha(value):
    return type(value) is str and DIGEST.fullmatch(value) is not None


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def gate(rows, name, result):
    rows.append({"name": name, "passed": result is True})


def normalize_config(config):
    value = deepcopy(config)
    value.pop("jobs_dir", None)
    value.pop("job_name", None)
    for agent in value.get("agents", []):
        agent.get("kwargs", {}).pop("semantic_transport_schema", None)
    return value


def normalize_trial_config(config):
    value = deepcopy(config)
    for key in ("trial_name", "trials_dir", "job_id"):
        value.pop(key, None)
    mapping(value.get("agent")).get("kwargs", {}).pop("semantic_transport_schema", None)
    return value


def cache_projection(cache):
    """Compare advisory work and pinned assets, not transient available memory."""
    cache = mapping(cache)
    result = {key: cache.get(key) for key in (
        "schema", "policy", "selection", "archive_sha256", "completed", "phase",
        "global_drop_caches", "freed_bytes_claimed", "credential_contents_recorded")}
    resource = mapping(cache.get("resources"))
    result["resource_limits"] = {key: resource.get(key) for key in (
        "schema", "cpu_max", "memory_max", "detected_cpu_slots", "detected_total_memory_mb")}
    for key in ("archive", "native_binaries", "native_libraries"):
        component = mapping(cache.get(key))
        result[key] = {name: component.get(name) for name in (
            "schema", "archive_sha256", "manifest_sha256", "codex_version", "selected_bytes",
            "selected_files", "advised_bytes", "advised_files", "body_read_bytes", "body_reads",
            "body_reads_after_advice", "body_writes", "metadata_unchanged", "global_drop_caches",
            "cgroup_writes", "errors")}
        files = component.get("files")
        result[key]["artifact_inventory"] = [{name: item.get(name) for name in (
            "path", "name", "role", "bytes", "sha256", "mode", "uid")}
            for item in files if type(item) is dict] if type(files) is list else None
    return result


def load_reuse(checkout):
    """Load the existing metadata comparator without executing any run path."""
    checkout = Path(checkout).absolute()
    if checkout.resolve(strict=True) != checkout or not checkout.is_dir():
        raise ValueError("canonical source checkout required")
    sys.path.insert(0, str(checkout))
    from benchmarks.agent_supervisor.container_coding import benchmark_controls as controls
    from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
    bindings = []
    for module in (controls, comparison):
        path = Path(module.__file__).absolute()
        path.relative_to(checkout)
        raw = path.read_bytes()
        bindings.append({"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    return controls, comparison, bindings


def arm(directory, expected_transport, controls, comparison, *, require_planning_schema=False,
        planning_source_bindings=()):
    directory = Path(directory).absolute()
    if directory.resolve(strict=True) != directory or not directory.is_dir():
        raise ValueError("canonical arm directory required")
    bindings, gates = [], []
    receipt, receipt_binding = load(directory / "receipt.json", bindings)
    prepared, _ = load(directory / "preparation.json", bindings)
    config, config_binding = load(directory / "config.json", bindings)
    gate(gates, "supported_receipt_schema", receipt.get("schema") == comparison.SUPERVISOR)
    gate(gates, "complete_single_trial_receipt", receipt.get("complete_single_trial_receipt") is True)
    trials = receipt.get("trials")
    if type(trials) is not list or len(trials) != 1 or receipt.get("trial_count") != 1:
        raise ValueError("exactly one retained trial required; do not silently drop an attempt")
    trial = trials[0]
    if type(trial) is not dict or type(trial.get("trial")) is not str:
        raise ValueError("native retained trial identity required")
    trial_name = trial["trial"]
    if Path(trial_name).name != trial_name or trial_name in (".", ".."):
        raise ValueError("bounded native trial name required")
    supervisor = mapping(trial.get("supervisor"))
    metadata = mapping(mapping(trial.get("agent_context")).get("metadata"))
    agent = config.get("agents", [{}])[0]
    kwargs = mapping(mapping(agent).get("kwargs"))
    gate(gates, "one_full_supervisor_agent", type(config.get("agents")) is list
        and len(config["agents"]) == 1 and receipt.get("arm") == kwargs.get("arm") == "full")
    gate(gates, "prepared_configuration_unchanged", prepared.get("prepared") is True
        and prepared.get("config_sha256") == config_binding["sha256"])
    observed_controls = mapping(receipt.get("comparison_controls"))
    declaration = mapping(observed_controls.get("declared"))
    gate(gates, "valid_frozen_controls", controls.validate_controls(declaration))
    gate(gates, "observed_configuration_unchanged", observed_controls.get("status") == "observed"
        and observed_controls.get("configuration_unchanged") is True)
    gate(gates, "control_configuration_digest_matches", declaration.get("configuration_sha256") == digest(config))
    gate(gates, "preparation_controls_unchanged", declaration == prepared.get("comparison_controls"))
    identity = {key: receipt.get(key) for key in ("task", "model", "reasoning_effort", "cli_version")}
    gate(gates, "identity_matches_declaration_and_preparation", declaration.get("identity") == identity
        and all(prepared.get(key) == item for key, item in identity.items()))
    gate(gates, "original_task_inputs_unchanged", receipt.get("original_task_inputs_unchanged") is True
        and trial.get("exact_trial_task_matches") is True
        and declaration.get("task_input_sha256") == prepared.get("task_input_sha256"))
    gate(gates, "intent_contract_not_selected", all(receipt.get(key) is None and prepared.get(key) is None
        for key in ("intent_requirement_contract_cid", "intent_requirement_contract_sha256"))
        and kwargs.get("intent_requirement_contract") is None
        and receipt.get("intent_selection_config_unchanged") is True)
    gate(gates, "direct_planning_retained", all(item.get("planning_strategy") == "direct"
        for item in (receipt, prepared, metadata, mapping(supervisor.get("planning")))))
    transports = [item.get("semantic_transport_schema", TRANSPORTS[0]) for item in (receipt, prepared, kwargs)]
    gate(gates, "explicit_expected_transport", all(item == expected_transport for item in transports)
        and receipt.get("semantic_transport_config_unchanged", True) is True)
    gate(gates, "single_worker", receipt.get("parallel_workers") == 1)
    gate(gates, "planning_and_indexing_inside_agent_time", receipt.get("planning_and_cold_index_charged_to_agent_time") is True
        and metadata.get("planning_and_cold_index_charged_to_agent_time") is True)
    gate(gates, "runtime_archive_bound", sha(prepared.get("archive_sha256"))
        and prepared.get("archive_sha256") == metadata.get("runtime_archive_sha256")
        and sha(prepared.get("manifest_sha256")))
    if require_planning_schema:
        manifest, manifest_binding = load(Path(prepared["archive"]) / "manifest.json", bindings)
        gate(gates, "planning_generation_archive_manifest_unchanged",
            manifest_binding["sha256"] == prepared.get("manifest_sha256")
            and manifest.get("archive_sha256") == prepared.get("archive_sha256"))
        files = manifest.get("files", [])
        for source in planning_source_bindings:
            matches = [item for item in files if type(item) is dict
                and item.get("path") == "source/" + source["relative_path"]]
            gate(gates, "planning_generation_archive_source_" + source["relative_path"],
                len(matches) == 1 and matches[0].get("sha256") == source["sha256"])
    cache = mapping(metadata.get("setup_cache"))
    gate(gates, "setup_cache_advice_completed_and_bound", cache.get("completed") is True
        and cache.get("selection") == prepared.get("setup_cache_selection")
        and cache.get("archive_sha256") == prepared.get("archive_sha256")
        and cache.get("global_drop_caches") is False and cache.get("credential_contents_recorded") is False)
    row = comparison._trial(receipt, trial, source=receipt_binding)
    invocations = supervisor.get("provider_invocations", [])
    invocations = invocations if type(invocations) is list else []
    ids = [item.get("invocation_id") for item in invocations if type(item) is dict]
    gate(gates, "exactly_one_planning_and_one_coding_router_cli_session",
        len(invocations) == len(ids) == 2 and len(set(ids)) == 2
        and sorted(item.get("phase", "") for item in invocations) == ["coding", "planning"]
        and all(item.get("router_calls") == 1 and item.get("provider") == "codex_cli" for item in invocations))
    gate(gates, "provider_identity_matches", all(item.get("model") == identity["model"]
        and item.get("reasoning_effort") == identity["reasoning_effort"] for item in invocations))
    native_path = Path(config.get("jobs_dir", "")) / str(config.get("job_name", "")) / trial_name / "result.json"
    native_path.relative_to(directory)
    native, native_binding = load(native_path, bindings)
    gate(gates, "native_result_hash_bound", native_binding["sha256"] == trial.get("native_result_sha256"))
    native_config = mapping(native.get("config"))
    tasks = config.get("tasks", [])
    task = tasks[0] if type(tasks) is list and len(tasks) == 1 else {}
    gate(gates, "exact_native_task_binding", native.get("trial_name") == trial_name
        and native_config.get("trial_name") == trial_name
        and native_config.get("task") == task
        and mapping(native.get("task_id")).get("path") == mapping(task).get("path")
        and Path(str(mapping(task).get("path", ""))).name == identity["task"]
        and native.get("task_name") == trial.get("task"))
    gate(gates, "native_trial_controls_match_frozen_config", native_config.get("agent") == agent
        and all(native_config.get(key) == config.get(key) for key in native_config
            if key not in ("task", "trial_name", "trials_dir", "job_id", "source_trial", "agent")))
    gate(gates, "native_observations_unchanged", native.get("agent_result") == trial.get("agent_context")
        and mapping(native.get("verifier_result")).get("rewards") == trial.get("reward")
        and mapping(native.get("exception_info")).get("exception_type") == trial.get("exception_type"))
    job_path = native_path.parent.parent / "result.json"
    job, _ = load(job_path, bindings)
    gate(gates, "one_finished_native_job_trial", job.get("n_total_trials") == 1
        and bool(job.get("finished_at")) and receipt.get("native_job_result_present") is True)
    audit = mapping(supervisor.get("context_input_audit"))
    gate(gates, "complete_exact_coding_context_audit", audit.get("schema") == "terminal-final-context-audit@1"
        and audit.get("all_observed_coding_inputs_verified") is True
        and audit.get("any_native_input_verified") is True and audit.get("any_model_input_verified") is True
        and audit.get("coding_receipts_seen") == audit.get("usable_coding_receipts") == 1
        and audit.get("provider_calls") == 0 and audit.get("raw_prompts_exported") is False
        and audit.get("read_errors") == [])
    phases = []
    all_sessions_complete = bool(invocations)
    for invocation in invocations:
        native_usage = mapping(invocation.get("native_rollout_usage"))
        usage = mapping(native_usage.get("usage"))
        counters = {key: count(usage.get(key)) for key in TOKENS}
        arithmetic = (all(value is not None for value in counters.values())
            and counters["input_tokens"] + counters["output_tokens"] == counters["total_tokens"]
            and counters["cached_input_tokens"] <= counters["input_tokens"])
        complete = native_usage.get("task_complete_observed") is True and native_usage.get("usage_available") is True
        all_sessions_complete = all_sessions_complete and complete and arithmetic
        gate(gates, invocation.get("phase", "unknown") + "_native_usage_arithmetic", arithmetic)
        gate(gates, invocation.get("phase", "unknown") + "_complete_native_session_usage", complete
            and native_usage.get("cache_included_in_input") is True and sha(native_usage.get("rollout_sha256"))
            and count(native_usage.get("observed_token_count_records")) not in (None, 0))
        fields = {key: invocation.get(key) for key in (
            "native_prompt_bytes", "native_prompt_sha256", "router_prompt_bytes", "router_prompt_sha256",
            "model_prompt_bytes", "model_prompt_sha256", "workspace_advisory_bytes", "workspace_advisory_sha256")}
        gate(gates, invocation.get("phase", "unknown") + "_prompt_hashes_present", all(sha(fields[key])
            for key in ("native_prompt_sha256", "router_prompt_sha256", "model_prompt_sha256"))
            and all(count(fields[key]) is not None for key in (
                "native_prompt_bytes", "router_prompt_bytes", "model_prompt_bytes", "workspace_advisory_bytes")))
        translation = mapping(invocation.get("semantic_translation"))
        if invocation.get("phase") == "planning":
            gate(gates, "planning_prompt_unchanged_by_transport", not translation
                and fields["native_prompt_sha256"] == fields["router_prompt_sha256"] == fields["model_prompt_sha256"]
                and fields["native_prompt_bytes"] == fields["router_prompt_bytes"] == fields["model_prompt_bytes"]
                and fields["workspace_advisory_bytes"] == 0)
            if require_planning_schema:
                policy = mapping(mapping(invocation.get("provider_invocation_policy")).get("structured_output"))
                projection = mapping(policy.get("native_schema_projection"))
                observed = mapping(invocation.get("usage"))
                gate(gates, "planning_generation_schema_requested_and_validated",
                    policy.get("schema") == "codex-native-planning-json-schema@1"
                    and policy.get("native_schema_requested") is True
                    and policy.get("response_schema_validated") is True
                    and policy.get("plan_admitted") is False
                    and projection.get("projection_id") == "canonical-prompt-goal-codex-generation-schema@1")
                gate(gates, "planning_generation_projection_preserves_native_validation",
                    all(projection.get(key) is True for key in (
                        "top_level_id_omitted", "bounds_annotation_omitted", "definitions_relocated",
                        "canonical_validation_preserved", "shape_only"))
                    and all(projection.get(key) is False for key in (
                        "execution_authority", "proof_authority", "completion_authority",
                        "publication_authority", "scope_expansion_authority")))
                gate(gates, "planning_generation_schema_digests_bounded_and_bound",
                    all(sha(projection.get(key)) for key in (
                        "canonical_schema_sha256", "native_wire_schema_sha256"))
                    and all(count(projection.get(key)) is not None and 0 < projection[key] <= 65536
                        for key in ("canonical_schema_bytes", "native_wire_schema_bytes"))
                    and observed.get("codex_output_schema_sha256") == projection.get("native_wire_schema_sha256")
                    and type(observed.get("codex_output_schema_bytes")) is int
                    and observed.get("codex_output_schema_bytes") == projection.get("native_wire_schema_bytes"))
                gate(gates, "native_plan_independently_qualified_after_generation",
                    mapping(supervisor.get("planning")).get("qualified") is True)
                projected = {key: projection.get(key) for key in (
                    "projection_id", "canonical_schema_sha256", "canonical_schema_bytes",
                    "native_wire_schema_sha256", "native_wire_schema_bytes", "top_level_id_omitted",
                    "bounds_annotation_omitted", "definitions_relocated", "canonical_validation_preserved",
                    "shape_only", "execution_authority", "proof_authority", "completion_authority",
                    "publication_authority", "scope_expansion_authority")}
                fields["planning_generation"] = {"policy": {key: policy.get(key) for key in (
                    "schema", "native_schema_requested", "response_schema_validated", "plan_admitted")},
                    "native_schema_projection": projected,
                    "observed_schema_sha256": observed.get("codex_output_schema_sha256"),
                    "observed_schema_bytes": observed.get("codex_output_schema_bytes")}
        if invocation.get("phase") == "coding":
            version = expected_transport[-1]
            gate(gates, "correct_coding_transport_receipt", translation.get("schema") == "supervisor-semantic-router-encoding@" + version
                and translation.get("transport_schema", TRANSPORTS[0]) == expected_transport
                and translation.get("freshness_checked") is True
                and translation.get("native_prompt_sha256") == fields["native_prompt_sha256"]
                and translation.get("native_prompt_bytes") == fields["native_prompt_bytes"]
                and sha(translation.get("provider_prompt_sha256"))
                and translation.get("execution_authority") is False and translation.get("completion_authority") is False)
            doctor = mapping(invocation.get("doctor_residual_context"))
            instruction = mapping(invocation.get("public_instruction"))
            components = [count(translation.get("provider_prompt_bytes")), count(doctor.get("advisory_bytes")),
                count(instruction.get("block_bytes")), count(fields["workspace_advisory_bytes"])]
            gate(gates, "complete_appended_prompt_byte_accounting", all(item is not None for item in components)
                and sum(components[:3]) == fields["router_prompt_bytes"]
                and sum(components) == fields["model_prompt_bytes"])
            matches = [item for item in audit.get("matches", []) if type(item) is dict
                and item.get("invocation_id") == invocation.get("invocation_id")]
            audited = [item for item in matches if item.get("model_input_verified") is True
                and item.get("native_prompt_sha256") == fields["native_prompt_sha256"]
                and item.get("model_prompt_sha256") == fields["model_prompt_sha256"]
                and item.get("native_prompt_bytes") == fields["native_prompt_bytes"]
                and item.get("model_prompt_bytes") == fields["model_prompt_bytes"]
                and mapping(item.get("model_input_checks"))
                and all(value is True for value in item["model_input_checks"].values())]
            gate(gates, "coding_hashes_match_reconstructed_inputs", bool(audited))
            fields["prompt_components_bytes"] = dict(zip(("semantic_transport", "doctor", "public_instruction", "workspace"), components))
            fields["transport_receipt"] = {key: translation.get(key) for key in (
                "schema", "transport_schema", "provider_prompt_bytes", "provider_prompt_sha256", "translation_cid",
                "identifier_mappings", "identifier_occurrences", "freshness_checked")}
        phases.append({"phase": invocation.get("phase"), "invocation_id": invocation.get("invocation_id"),
            "status": invocation.get("status"), "complete_usage_observed": complete,
            "native_token_count_records": count(native_usage.get("observed_token_count_records")),
            "recorded_input": fields, "provider_controls": {key: invocation.get(key) for key in (
                "provider", "model", "reasoning_effort", "requested_output_tokens", "provider_output_token_cap_enforced", "timeout_seconds")}})
    usage_summary = mapping(metadata.get("usage"))
    gate(gates, "all_provider_attempts_receipted", usage_summary.get("all_invocations_receipted") is True
        and not supervisor.get("unreceipted_provider_attempt") and usage_summary.get("provider_calls") == 2)
    gate(gates, "native_total_equals_agent_summary", all(row["tokens"][key] is not None
        and row["tokens"][key] == usage_summary.get(key) for key in TOKENS)
        and usage_summary.get("cache_included_in_input") is True)
    gate(gates, "original_verifier_passed", row["official_reward"] == 1)
    gate(gates, "native_task_completed", supervisor.get("task_completed") is True)
    gate(gates, "worker_and_supervisor_cleaned_up", supervisor.get("remaining_processes") == 0
        and supervisor.get("worker_cleanup_returncode") == 0 and metadata.get("remaining_processes") == 0)
    gate(gates, "harbor_completed", receipt.get("harbor_returncode") == 0 and trial.get("exception_type") is None)
    if (not all_sessions_complete or usage_summary.get("all_invocations_receipted") is not True
            or usage_summary.get("provider_calls") != len(set(ids))
            or supervisor.get("unreceipted_provider_attempt")):
        row["tokens"] = {key: None for key in TOKENS}
        row["native_usage_complete_observed"] = False
    row["uncached_input_tokens"] = (row["tokens"]["input_tokens"] - row["tokens"]["cached_input_tokens"]
        if row["tokens"]["input_tokens"] is not None and row["tokens"]["cached_input_tokens"] is not None else None)
    return {"label": "baseline" if expected_transport == TRANSPORTS[0] else "compact", "transport": expected_transport,
        "row": row, "phases": phases, "qualification_checks": gates, "inputs": bindings,
        "normalized_configuration_sha256": digest(normalize_config(config)),
        "normalized_native_trial_configuration_sha256": digest(normalize_trial_config(native_config)),
        "observed_setup_cache_sha256": digest(cache_projection(metadata.get("setup_cache"))),
        "setup_cache_selection": prepared.get("setup_cache_selection"),
        "archive_sha256": prepared.get("archive_sha256"), "archive_manifest_sha256": prepared.get("manifest_sha256"),
        "host_sources_sha256": digest(prepared.get("host_source_sha256")),
        "source384_binding_sha256": digest(receipt.get("source384")),
        "intent_action_binding_sha256": digest(receipt.get("intent_action_384")),
        "resource_profile": receipt.get("resource_profile"),
        "execution_budget_sha256": digest(metadata.get("execution_budget")),
        "admission_environment_sha256": digest(metadata.get("admission_environment")),
        "planning_generation_schema_required": require_planning_schema,
        "declared_controls": receipt.get("comparison_controls")}


def compare(left, right, controls):
    gates = []
    common = controls.compare_controls(left["declared_controls"], right["declared_controls"])
    gate(gates, "same_declared_task_job_agent_controls", common.get("matches") is True)
    for field in ("normalized_configuration_sha256", "normalized_native_trial_configuration_sha256",
            "archive_sha256", "archive_manifest_sha256", "host_sources_sha256", "resource_profile",
            "source384_binding_sha256", "intent_action_binding_sha256", "setup_cache_selection",
            "observed_setup_cache_sha256", "execution_budget_sha256", "admission_environment_sha256"):
        gate(gates, "same_" + field, left[field] == right[field] and left[field] is not None)
    providers = lambda item: {row["phase"]: row["provider_controls"] for row in item["phases"]}
    gate(gates, "same_phase_provider_controls", providers(left) == providers(right))
    if left["planning_generation_schema_required"] or right["planning_generation_schema_required"]:
        generation = lambda item: [row["recorded_input"].get("planning_generation")
            for row in item["phases"] if row["phase"] == "planning"]
        gate(gates, "same_planning_generation_requirement", left["planning_generation_schema_required"] is True
            and right["planning_generation_schema_required"] is True)
        gate(gates, "same_requested_and_observed_planning_generation_schema", generation(left) == generation(right)
            and len(generation(left)) == 1 and generation(left)[0] is not None)
    gate(gates, "distinct_native_trials", left["row"]["trial"] != right["row"]["trial"])
    qualified = all(item["passed"] for item in (*left["qualification_checks"], *right["qualification_checks"], *gates))
    observed_delta = {}
    for key in TOKENS:
        a, b = left["row"]["tokens"][key], right["row"]["tokens"][key]
        observed_delta[key] = a - b if a is not None and b is not None else None
    total = left["row"]["tokens"]["total_tokens"]
    total_delta = observed_delta["total_tokens"]
    percentage = 100 * total_delta / total if total not in (None, 0) and total_delta is not None and qualified else None
    return {"qualification_checks": gates, "declared_controls_comparison": common,
        "matched_pair_qualified": qualified, "observed_baseline_minus_compact_tokens": observed_delta,
        "qualified_pair_observed_total_token_reduction_percent": percentage,
        "repeatable_or_causal_reduction_established": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-qualified", action="store_true")
    parser.add_argument("--require-planning-schema", action="store_true",
        help="Require canonical Codex generation schema custody in both new arms.")
    args = parser.parse_args()
    if args.output.exists() or args.output.absolute().resolve() != args.output.absolute():
        raise ValueError("fresh canonical output required")
    controls, comparison, sources = load_reuse(args.source_checkout)
    planning_sources = []
    if args.require_planning_schema:
        for relative in ("ipfs_accelerate_py/llm_router.py",
                "ipfs_accelerate_py/agent_supervisor/runtime/codex_planning_schema.py",
                "ipfs_accelerate_py/agent_supervisor/runtime/router_implementation_runner.py",
                "ipfs_accelerate_py/cli_runtime/grok_structured_output.py",
                "ipfs_accelerate_py/agent_supervisor/prompt/prompt_goal_planner.py"):
            path = args.source_checkout.absolute() / relative
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 2 * 1024 * 1024:
                raise ValueError("bounded regular planning generation source required")
            raw = path.read_bytes()
            planning_sources.append({"relative_path": relative, "path": str(path),
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    left = arm(args.baseline, TRANSPORTS[0], controls, comparison,
        require_planning_schema=args.require_planning_schema, planning_source_bindings=planning_sources)
    right = arm(args.compact, TRANSPORTS[1], controls, comparison,
        require_planning_schema=args.require_planning_schema, planning_source_bindings=planning_sources)
    result = compare(left, right, controls)
    payload = {"schema": "terminal-planning-retained-transport-pair-accounting@1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reused_comparison_sources": sources, "arms": [left, right], "comparison": result,
        "planning_generation_source_bindings": planning_sources,
        "new_provider_calls_by_accounting": 0, "raw_model_data_read": False,
        "cached_input_already_in_input_total": True, "unknown_fields_are_zero": False,
        "benchmark_advantage_claimed": False, "provider_cost_calculated": False,
        "limitations": [
            "One matched pair is descriptive; it does not establish a repeatable or causal saving percentage.",
            "Internal model turns, output length, tool usage and stochastic choices can differ despite matching declared controls.",
            "Initial model input hashes exclude provider harness/system context and later internal history; native cumulative tokens include the observed complete sessions.",
            "The existing context audit reconstructs coding inputs; planning prompt consistency is checked from receipts without an independent planning reconstruction.",
            "Matching setup advice and selected asset inventories does not prove equal kernel or provider cache occupancy.",
            "Recorded native controls are checked against frozen configurations; this producer does not independently prove runtime enforcement.",
            "Cached input is already counted in input tokens; cache/reasoning subcategories are not added again.",
            "Failed and interrupted attempts remain visible; incomplete totals are unknown and observed counters are retained as subtotals.",
            "No transcript, model thought, task source body, solution or hidden verifier body is read; only bounded metadata and comparator/planning generation implementation sources are accessed.",
        ]}
    with args.output.open("x") as stream:
        stream.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output.absolute()), "matched_pair_qualified": result["matched_pair_qualified"],
        "observed_baseline_minus_compact_total_tokens": result["observed_baseline_minus_compact_tokens"]["total_tokens"],
        "repeatable_or_causal_reduction_established": False}))
    if args.require_qualified and not result["matched_pair_qualified"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
