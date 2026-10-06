"""Bounded metadata comparison of one symbolic, direct, and native trial.

Reads retained receipts, preparations, configurations, and signed admission
envelopes only. It never opens model transcripts, solution bodies, credentials,
or hidden verifier source. Existing outputs are refused. Counter owners are
loaded from exact Git source and the archived measured_usage owner is checked
against the original bundle manifest. CLI/router sessions are not API turns.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess

MAX_BYTES = 64 * 1024 * 1024
TOKENS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
PHASES = ("environment_setup", "agent_setup", "agent_execution", "verifier")
OWNER = "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py"
COMPARISON = "benchmarks/agent_supervisor/container_coding/benchmark_comparison.py"
CONTROLS = "benchmarks/agent_supervisor/container_coding/benchmark_controls.py"
SOURCE384_CONSUMER = "ipfs_accelerate_py/agent_supervisor/runtime/source384_repository_context.py"
AUTHORITY = ("source_semantics_verified", "semantic_alignment_verified", "proof_authority",
             "execution_authority", "completion_authority")
EXPECTED_ARCHIVE = "a9f25619d230725ebe9f0f784d74feaf9191905ab78216fcedaee45a95c0cbcc"
EXPECTED_RUNNER = "8dab684ef69f87a88c3dac1585eea07a5bf29a24"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate metadata key")
        result[key] = value
    return result


def read(path, *, limit=MAX_BYTES):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path:
        raise ValueError("canonical regular metadata input required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("bounded regular metadata input required")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (len(raw) > limit or (before.st_size, before.st_mtime_ns) !=
            (after.st_size, after.st_mtime_ns)):
        raise ValueError("metadata input changed during inspection")
    return raw, {"path": str(path), "bytes": len(raw), "sha256": sha(raw)}


def load(path, evidence):
    raw, binding = read(path)
    value = json.loads(raw, object_pairs_hook=strict_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite metadata input")))
    if type(value) is not dict:
        raise ValueError("metadata object required")
    evidence.append(binding)
    return value


def obj(value):
    return value if type(value) is dict else {}


def number(value, *, integer=False):
    valid = type(value) is int if integer else type(value) in (int, float)
    return value if valid and math.isfinite(value) and 0 <= value <= 2**63 - 1 else None


def boolean(value):
    return value if type(value) is bool else None


def identity(value):
    return value if type(value) is str and re.fullmatch(r"[A-Za-z0-9_.@+-]{1,128}", value) else None


def digest(value):
    return value if type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) else None


def closed(value, allowed):
    return None if value is None else value if type(value) is str and value in allowed else "unrecognized"


def text_hash(value):
    return sha(value.encode()) if type(value) is str and len(value) <= 1024 else None


def token_counts(value):
    return {key: number(obj(value).get(key), integer=True) for key in TOKENS}


def sum_known(values):
    values = list(values)
    return sum(values) if values and all(value is not None for value in values) else None


def command_arg(recipe, flag):
    argv = recipe.get("argv")
    if type(argv) is not list or argv.count(flag) != 1:
        raise ValueError("exact selected build argument required")
    value = argv[argv.index(flag) + 1]
    if type(value) is not str:
        raise ValueError("selected build argument must be text")
    return value


def git_source(checkout, head, relative, evidence):
    if type(head) is not str or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("exact source revision required")
    raw = subprocess.check_output(["git", "-C", str(checkout), "show", head + ":" + relative], timeout=30)
    if len(raw) > 512 * 1024:
        raise ValueError("bounded source owner required")
    evidence.append({"git_head": head, "path": relative, "bytes": len(raw), "sha256": sha(raw)})
    return raw


def counter_owners(checkout, head, build, manifest, evidence):
    """Execute only source-defined counter functions, never runtime entrypoints."""
    source_root = command_arg(build, "--source")
    source_head = obj(build.get("source_heads")).get(source_root)
    owner_raw = git_source(source_root, source_head, OWNER, evidence)
    matches = [row for row in manifest.get("files", []) if type(row) is dict
               and row.get("path") == "source/" + OWNER]
    if len(matches) != 1 or matches[0].get("sha256") != sha(owner_raw):
        raise ValueError("measured_usage source differs from exact runtime bundle")
    source384_raw = git_source(source_root, source_head, SOURCE384_CONSUMER, evidence)
    consumer_rows = [row for row in manifest.get("files", []) if type(row) is dict
                     and row.get("path") == "source/" + SOURCE384_CONSUMER]
    if len(consumer_rows) != 1 or consumer_rows[0].get("sha256") != sha(source384_raw):
        raise ValueError("Source384 consumer differs from the exact runtime bundle")
    owner_nodes = [node for node in ast.parse(owner_raw).body
                   if isinstance(node, ast.FunctionDef) and node.name == "measured_usage"]
    if len(owner_nodes) != 1:
        raise ValueError("one archived measured_usage function required")
    owner_namespace = {}
    exec(compile(ast.Module(body=owner_nodes, type_ignores=[]), OWNER, "exec"), owner_namespace)

    controls_raw = git_source(checkout, head, CONTROLS, evidence)
    comparison_raw = git_source(checkout, head, COMPARISON, evidence)
    for relative, frozen in ((CONTROLS, controls_raw), (COMPARISON, comparison_raw)):
        current, _ = read(Path(checkout) / relative, limit=512 * 1024)
        if current != frozen:
            raise ValueError("host reporting owner differs from selected Git source")
    namespace = {"hashlib": hashlib, "html": html, "json": json, "math": math,
                 "Path": Path, "re": re, "__file__": str(Path(checkout) / COMPARISON)}
    control_names = {"_digest", "validate_controls", "compare_controls"}
    control_assignments = {"SCHEMA", "_DIGEST", "JOB_KEYS", "AGENT_KEYS"}
    nodes = [node for node in ast.parse(controls_raw).body if
             isinstance(node, ast.FunctionDef) and node.name in control_names or
             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                 and t.id in control_assignments for t in node.targets)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), CONTROLS, "exec"), namespace)
    comparison_functions = []
    for node in ast.parse(comparison_raw).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_cell":
            break
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            comparison_functions.append(node)
    exec(compile(ast.Module(body=comparison_functions, type_ignores=[]), COMPARISON, "exec"), namespace)
    return owner_namespace["measured_usage"], namespace["collect"], {
        "measured_usage_sha256": sha(owner_raw), "comparison_sha256": sha(comparison_raw),
        "controls_sha256": sha(controls_raw), "source384_consumer_sha256": sha(source384_raw)}


def one_trial(receipt, schema):
    rows = receipt.get("trials")
    if receipt.get("schema") != schema or receipt.get("trial_count") != 1 or type(rows) is not list or len(rows) != 1:
        raise ValueError("one retained trial per selected arm required")
    if identity(rows[0].get("trial")) is None or receipt.get("task") != "largest-eigenval":
        raise ValueError("selected original task identity required")
    return rows[0]


def provider_rows(report):
    rows = report.get("provider_invocations", [])
    if type(rows) is not list or len(rows) > 64 or any(type(row) is not dict for row in rows):
        raise ValueError("bounded provider receipt inventory required")
    found = {}
    for row in rows:
        key = row.get("invocation_id")
        if type(key) is not str or not key or len(key) > 256:
            raise ValueError("provider invocation identity required")
        if key in found and found[key] != row:
            raise ValueError("conflicting provider invocation identity")
        found[key] = row
    return list(found.values())


def invocation(row):
    native = obj(row.get("native_rollout_usage"))
    completed = boolean(native.get("task_complete_observed"))
    subtotal = token_counts(native.get("usage"))
    return {"invocation_id_sha256": text_hash(row.get("invocation_id")),
        "phase": closed(row.get("phase", row.get("purpose")), {"planning", "coding"}),
        "status": closed(row.get("status"), {"provider_returned", "failed"}),
        "provider": closed(row.get("provider"), {"codex_cli", "grok_cli"}),
        "model": identity(row.get("model")), "reasoning_effort": identity(row.get("reasoning_effort")),
        "router_calls": number(row.get("router_calls"), integer=True),
        "seconds": number(row.get("seconds")),
        "timeout_seconds": number(row.get("timeout_seconds")),
        "native_token_count_records": number(native.get("observed_token_count_records"), integer=True),
        "native_task_complete_observed": completed,
        "tokens": subtotal if completed is True else {key: None for key in TOKENS},
        "known_token_subtotals": subtotal,
        "native_rollout_sha256": digest(native.get("rollout_sha256")),
        "native_session_id_sha256": text_hash(native.get("thread_id")),
        "input_binding_sha256": digest(row.get("native_prompt_sha256")),
        "model_input_binding_sha256": digest(row.get("model_prompt_sha256")),
        "cash_cost": None, "billing_total_verified": False}


def coverage_projection(value):
    value = obj(value)
    def count_list(key):
        rows = value.get(key)
        return len(rows) if type(rows) is list and len(rows) <= 256 else None
    return {"accepted": boolean(value.get("accepted")),
        "bindings": count_list("bindings"), "uncovered_requirements": count_list("uncovered_requirement_ids"),
        "unsupported_requirements": count_list("unsupported_requirement_ids"),
        "errors": count_list("errors"), "prohibited_output_violations": count_list("prohibited_output_violations"),
        "source_accounting_complete": boolean(value.get("source_accounting_complete")),
        "semantic_support_complete": boolean(value.get("semantic_support_complete")),
        "ledger_sha256": digest(value.get("ledger_sha256")),
        **{key: boolean(value.get(key)) for key in AUTHORITY if key != "source_semantics_verified"}}


def planning_projection(report):
    planning = obj(report.get("planning"))
    symbolic = obj(planning.get("symbolic_planning"))
    return {"strategy": closed(planning.get("planning_strategy"), {"direct", "intent_symbolic", "intent_coverage"}),
        "qualified": boolean(planning.get("qualified")),
        "provider_calls": number(planning.get("provider_calls"), integer=True),
        "goals": number(planning.get("goals"), integer=True), "tasks": number(planning.get("tasks"), integer=True),
        "elapsed_seconds": number(planning.get("elapsed_seconds")),
        "requirement_coverage": coverage_projection(planning.get("requirement_coverage")) if "requirement_coverage" in planning else None,
        "symbolic": {"accepted": boolean(symbolic.get("accepted")),
            "interpretation_scope": closed(symbolic.get("interpretation_scope"), {"administrative_requirement_task_coverage"}),
            "provider_calls": number(symbolic.get("provider_calls"), integer=True),
            "critic_decision": closed(symbolic.get("critic_decision"), {"accepted", "rejected"}),
            **{key: boolean(symbolic.get(key)) for key in AUTHORITY},
            "receipt_sha256": sha(canonical(symbolic))} if symbolic else None,
        **{key: boolean(planning.get(key)) for key in AUTHORITY}}


def source384_projection(report, metadata):
    initial = obj(report.get("initial_context"))
    source = obj(initial.get("source384_context"))
    assets = obj(metadata.get("source384_assets"))
    summary = obj(source.get("summary"))
    resources = obj(obj(metadata.get("setup_cache")).get("resources"))
    return {"enabled": boolean(assets.get("enabled")), "selected": boolean(assets.get("selected")),
        "neural_inference_replayed": boolean(source.get("neural_inference_replayed")),
        "checkpoint_sha256": digest(source.get("checkpoint_sha256")),
        "config_sha256": digest(source.get("config_sha256")),
        "inference_sha256": digest(source.get("inference_sha256")),
        "seconds": number(source.get("seconds")),
        "training_steps": number(source.get("training_steps"), integer=True),
        "download_calls": number(assets.get("download_calls"), integer=True),
        "summary": {key: number(summary.get(key), integer=True) for key in
            ("source_files", "program_source_files", "harness_support_files", "inference_python_files", "omitted_candidates", "provider_calls")},
        "formalization_authority": boolean(source.get("formalization_authority")),
        "proof_authority": boolean(source.get("proof_authority")),
        "initial_context_provider_calls": number(initial.get("provider_calls"), integer=True),
        "initial_context_learned_embeddings": boolean(initial.get("learned_embeddings")),
        "source384_nomination_only": boolean(summary.get("nomination_only")),
        "candidate_status_counts": {key: number(value, integer=True)
            for key, value in obj(obj(summary.get("coverage")).get("candidate_statuses")).items()
            if key in {"fail_open_source_contract_unsupported", "source_qualified", "unsupported"}},
        "observed_cgroup_cpu_slots": number(resources.get("detected_cpu_slots"), integer=True),
        "observed_cgroup_memory_mb": number(resources.get("detected_total_memory_mb"), integer=True)}


def supervisor_projection(receipt, trial, comparison, measured_usage):
    report = obj(trial.get("supervisor"))
    metadata = obj(obj(trial.get("agent_context")).get("metadata"))
    invocations = [invocation(row) for row in provider_rows(report)]
    phases = obj(report.get("phases"))
    observed = measured_usage(report)
    tokens = token_counts(comparison.get("tokens"))
    subtotal = token_counts(comparison.get("known_token_subtotals"))
    if observed.get("all_invocations_receipted") is True and token_counts(observed) != subtotal:
        raise ValueError("frozen usage owner and native receipt subtotal differ")
    return {"trial": identity(trial.get("trial")), "task": "largest-eigenval", "arm": "full",
        "declared_model": identity(receipt.get("model")), "declared_cli_version": identity(receipt.get("cli_version")),
        "declared_reasoning_effort": identity(receipt.get("reasoning_effort")),
        "official_reward": number(comparison.get("official_reward")),
        "exception_type": identity(comparison.get("exception_type")),
        "native_task_completed": boolean(report.get("task_completed")),
        "usage_complete_observed": boolean(comparison.get("native_usage_complete_observed")),
        "all_provider_attempts_receipted": boolean(observed.get("all_invocations_receipted")),
        "router_sessions": number(comparison.get("provider_calls"), integer=True),
        "observed_router_receipts": len(invocations),
        "tokens": tokens, "known_token_subtotals": subtotal,
        "native_token_count_records": sum_known(row["native_token_count_records"] for row in invocations),
        "provider_invocations": invocations,
        "planning": planning_projection(report),
        "source384": source384_projection(report, metadata),
        "durations_seconds": {key: number(obj(trial.get("durations_seconds")).get(key)) for key in PHASES},
        "supervisor_phase_seconds": {key: number(phases.get(key)) for key in
            ("prepare_seconds", "initial_context_seconds", "planning_seconds", "context_seconds", "doctor_seconds", "post_publication_refresh_seconds")},
        "coding_provider_seconds": sum_known(row["seconds"] for row in invocations if row["phase"] == "coding"),
        "planning_provider_seconds": sum_known(row["seconds"] for row in invocations if row["phase"] == "planning")
            if any(row["phase"] == "planning" for row in invocations) else 0 if planning_projection(report)["provider_calls"] == 0 else None,
        "cleanup_seconds": None,
        "cleanup_duration_instrumented": False,
        "cleanup_returncode": number(report.get("worker_cleanup_returncode"), integer=True),
        "remaining_processes": number(report.get("remaining_processes"), integer=True),
        "stop_status": closed(obj(report.get("stop")).get("status"), {"succeeded", "failed", "timed_out", "unavailable"}),
        "supervisor_seconds": number(report.get("seconds")),
        "outer_invocation_seconds": number(receipt.get("invocation_seconds")),
        "runtime_archive_sha256": digest(metadata.get("runtime_archive_sha256")),
        "original_task_inputs_unchanged": boolean(receipt.get("original_task_inputs_unchanged")),
        "signed_admission_exported": boolean(metadata.get("signed_admission_exported")),
        "implementation_route": closed(report.get("implementation_route"), {"model_router", "doctor", "native_doctor"}),
        "billing_total_verified": False, "dollar_cost": None}


def native_projection(receipt, trial, comparison):
    native = obj(trial.get("raw_usage"))
    sessions = native.get("sessions")
    if type(sessions) is not list or not 1 <= len(sessions) <= 64:
        raise ValueError("bounded retained native session inventory required")
    projected = [{"session_id_sha256": text_hash(row.get("session_id")),
        "rollout_sha256": digest(row.get("sha256")), "cli_version": identity(row.get("cli_version")),
        "observed_models": [identity(v) for v in row.get("observed_models", [])],
        "observed_reasoning_efforts": [identity(v) for v in row.get("observed_reasoning_efforts", [])],
        "task_complete_observed": boolean(row.get("task_complete_observed")),
        "native_token_count_records": number(row.get("token_count_events"), integer=True),
        "usage": token_counts(row.get("usage"))} for row in sessions if type(row) is dict]
    if len(projected) != len(sessions):
        raise ValueError("native session metadata object required")
    return {"trial": identity(trial.get("trial")), "task": "largest-eigenval", "arm": "native-codex",
        "official_reward": number(comparison.get("official_reward")),
        "usage_complete_observed": boolean(comparison.get("native_usage_complete_observed")),
        "cli_sessions": len(projected), "sessions": projected,
        "native_token_count_records": sum_known(row["native_token_count_records"] for row in projected),
        "tokens": token_counts(comparison.get("tokens")),
        "known_token_subtotals": token_counts(comparison.get("known_token_subtotals")),
        "durations_seconds": {key: number(obj(trial.get("seconds")).get(key)) for key in PHASES},
        "outer_invocation_seconds": number(receipt.get("invocation_seconds")),
        "original_task_inputs_unchanged": boolean(receipt.get("original_task_inputs_unchanged")),
        "billing_total_verified": False, "dollar_cost": None}


def normalize_config(config, *, remove_contract):
    config = json.loads(json.dumps(config))
    for key in ("job_name", "jobs_dir"):
        config.pop(key, None)
    retry = obj(config.get("retry"))
    if retry.get("max_retries") == 0:
        for key in ("exclude_exceptions", "include_exceptions"):
            if type(retry.get(key)) is list:
                retry[key] = sorted(retry[key])
    if remove_contract:
        config["agents"][0]["kwargs"].pop("intent_requirement_contract", None)
    return config


def require_prepared(receipt, prepared, config_sha256, manifest):
    for key in ("task", "model", "reasoning_effort", "cli_version", "resource_profile", "planning_strategy",
                "intent_requirement_contract_sha256", "intent_requirement_contract_cid"):
        if receipt.get(key) != prepared.get(key):
            raise ValueError("runtime identity differs from exact preparation: " + key)
    for key in ("model", "reasoning_effort", "cli_version"):
        if identity(receipt.get(key)) is None:
            raise ValueError("closed provider identity required")
    if (prepared.get("archive_sha256") != EXPECTED_ARCHIVE or manifest.get("archive_sha256") != EXPECTED_ARCHIVE
            or manifest.get("codex_version") != receipt.get("cli_version")):
        raise ValueError("prepared frozen runtime or CLI differs")
    if prepared.get("config_sha256") != config_sha256:
        raise ValueError("configuration differs from frozen preparation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--runner-head", default=EXPECTED_RUNNER)
    parser.add_argument("--symbolic", type=Path, required=True, help="completed run directory")
    parser.add_argument("--direct", type=Path, required=True, help="historical direct run directory")
    parser.add_argument("--baseline", type=Path, required=True, help="native receipt.json")
    parser.add_argument("--build-command", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--runtime-observation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("fresh bounded comparison output required")
    if args.runner_head != EXPECTED_RUNNER:
        raise ValueError("selected trial's exact frozen host runner required")
    evidence = []
    symbolic_receipt = load(args.symbolic / "receipt.json", evidence)
    direct_receipt = load(args.direct / "receipt.json", evidence)
    native_receipt = load(args.baseline, evidence)
    symbolic_prepared = load(args.symbolic / "preparation.json", evidence)
    direct_prepared = load(args.direct / "preparation.json", evidence)
    symbolic_config = load(args.symbolic / "config.json", evidence)
    symbolic_config_sha256 = evidence[-1]["sha256"]
    direct_config = load(args.direct / "config.json", evidence)
    direct_config_sha256 = evidence[-1]["sha256"]
    native_config = load(args.baseline.parent / "config.json", evidence)
    build = load(args.build_command, evidence)
    contract = load(args.contract, evidence)
    archive = Path(symbolic_prepared["archive"])
    if archive != Path(direct_prepared["archive"]) or archive != Path(command_arg(build, "--output")):
        raise ValueError("both arms must use the original selected archive")
    manifest = load(archive / "manifest.json", evidence)
    manifest_sha = evidence[-1]["sha256"]
    if any(prepared.get("manifest_sha256") != manifest_sha for prepared in (symbolic_prepared, direct_prepared)):
        raise ValueError("original frozen manifest binding differs")
    require_prepared(symbolic_receipt, symbolic_prepared, symbolic_config_sha256, manifest)
    require_prepared(direct_receipt, direct_prepared, direct_config_sha256, manifest)
    if (contract.get("schema") != "intent-plan-requirement-contract@2" or
            symbolic_config["agents"][0]["kwargs"].get("intent_requirement_contract") != contract or
            sha(canonical(contract)) != symbolic_prepared.get("intent_requirement_contract_sha256")):
        raise ValueError("prepared symbolic contract differs from private source-bound candidate")
    measured_usage, collect, source_owners = counter_owners(args.checkout, args.runner_head, build, manifest, evidence)
    st = one_trial(symbolic_receipt, "terminal-full-supervisor-receipt@1")
    dt = one_trial(direct_receipt, "terminal-full-supervisor-receipt@1")
    nt = one_trial(native_receipt, "native-codex-harbor-baseline-receipt@1")
    comparison = collect(baseline=args.baseline, supervisors=[args.direct / "receipt.json", args.symbolic / "receipt.json"])
    if len(comparison["rows"]) != 3:
        raise ValueError("three independent retained trials required")
    selected = {row["trial"]: row for row in comparison["rows"]}
    symbolic = supervisor_projection(symbolic_receipt, st, selected[st["trial"]], measured_usage)
    direct = supervisor_projection(direct_receipt, dt, selected[dt["trial"]], measured_usage)
    native = native_projection(native_receipt, nt, selected[nt["trial"]])
    for trial in (st, dt):
        consumer = obj(obj(obj(trial.get("supervisor")).get("initial_context")).get("source384_context")).get("producer", {}).get("consumer")
        if consumer != source_owners["source384_consumer_sha256"]:
            raise ValueError("reported Source384 consumer differs from archived source")
    # Actual official result bytes corroborate receipt identity and reward,
    # without following any hidden verifier artifacts or reference paths.
    original_results = []
    admissions = []
    for run, trial, receipt, row in ((args.symbolic, st, symbolic_receipt, symbolic),
                                   (args.direct, dt, direct_receipt, direct)):
        result_path = run / "jobs" / "supervisor-full-largest-eigenval" / trial["trial"] / "result.json"
        result = load(result_path, evidence)
        if evidence[-1]["sha256"] != trial.get("native_result_sha256") or obj(result.get("verifier_result")).get("rewards") != trial.get("reward"):
            raise ValueError("official supervisor result differs from canonical receipt")
        original_results.append({"trial": trial["trial"], "finished": bool(result.get("finished_at")),
                                 "reward_matches": True})
        admission_path = result_path.parent / "agent" / "admission.json"
        if row["signed_admission_exported"] is True:
            admission = load(admission_path, evidence)
            payload = obj(obj(admission.get("receipt")).get("payload"))
            admitted_coverage = obj(payload.get("requirement_coverage"))
            planning_coverage = obj(obj(trial.get("supervisor")).get("planning")).get("requirement_coverage")
            if row is symbolic and admitted_coverage != planning_coverage:
                raise ValueError("native signed admission differs from observed planning coverage")
            admissions.append({"trial": trial["trial"], "envelope_sha256": evidence[-1]["sha256"],
                "planning_permitted": boolean(payload.get("planning_permitted")),
                "code_proof_authority": boolean(payload.get("code_proof_authority")),
                "completion_authority": boolean(payload.get("completion_authority")),
                "production_activation": boolean(payload.get("production_activation")),
                "coverage": coverage_projection(admitted_coverage) if admitted_coverage else None,
                "cryptographic_replay_performed_here": False})
    native_result_path = args.baseline.parent / "jobs" / "native-codex-largest-eigenval" / nt["trial"] / "result.json"
    native_result = load(native_result_path, evidence)
    if evidence[-1]["sha256"] != nt.get("result_sha256") or obj(native_result.get("verifier_result")).get("rewards") != nt.get("reward"):
        raise ValueError("original native result differs from canonical receipt")

    prepared_inputs = symbolic_prepared.get("task_input_sha256")
    controls = {label: obj(receipt.get("comparison_controls")) for label, receipt in
                (("symbolic", symbolic_receipt), ("direct", direct_receipt), ("native", native_receipt))}
    retries_disabled = all(config.get("retry", {}).get("max_retries") == 0 for config in
                           (symbolic_config, direct_config, native_config))
    requirement_count = len(contract.get("requirements", []))
    atoms = obj(contract.get("ledger")).get("requirements", [])
    if type(atoms) is not list or len(atoms) != requirement_count:
        raise ValueError("candidate requirement inventory differs")
    ledger = obj(contract.get("ledger"))
    operation_contract = obj(contract.get("symbolic_operations"))
    runtime_observation = load(args.runtime_observation, evidence) if args.runtime_observation else {}
    # This already bounded observation is hashed, but its arbitrary fields are
    # never copied into the publication. Exact container fingerprints differ
    # historically even when the source archive is reused.
    runtime_observation_binding = evidence[-1] if args.runtime_observation else None
    checks = {
        "all_original_task_inputs_unchanged": all(row["original_task_inputs_unchanged"] is True for row in (symbolic, direct, native)),
        "same_public_and_verifier_input_hash_inventory": prepared_inputs == direct_prepared.get("task_input_sha256") == obj(controls["native"].get("declared")).get("task_input_sha256"),
        "same_frozen_supervisor_archive_observed": symbolic["runtime_archive_sha256"] == direct["runtime_archive_sha256"] == EXPECTED_ARCHIVE,
        "same_frozen_source384_checkpoint_config": symbolic["source384"]["checkpoint_sha256"] == direct["source384"]["checkpoint_sha256"] and symbolic["source384"]["config_sha256"] == direct["source384"]["config_sha256"],
        "same_provider_identity_declared": all(tuple(receipt.get(k) for k in ("task", "model", "reasoning_effort", "cli_version")) == tuple(native_receipt.get(k) for k in ("task", "model", "reasoning_effort", "cli_version")) for receipt in (symbolic_receipt, direct_receipt)),
        "symbolic_actual_invocations_match_declared_identity": all(row["model"] == symbolic_receipt["model"] and row["reasoning_effort"] == symbolic_receipt["reasoning_effort"] for row in symbolic["provider_invocations"]),
        "prepared_strategy_is_intent_symbolic": symbolic_prepared.get("planning_strategy") == "intent_symbolic",
        "actual_symbolic_planning_zero_provider_calls": symbolic["planning"]["provider_calls"] == 0 and not any(row["phase"] == "planning" for row in symbolic["provider_invocations"]),
        "actual_symbolic_all_eight_requirement_atoms_covered": requirement_count == 8 and symbolic["planning"]["requirement_coverage"] is not None and symbolic["planning"]["requirement_coverage"]["accepted"] is True and symbolic["planning"]["requirement_coverage"]["bindings"] == 8 and symbolic["planning"]["requirement_coverage"]["uncovered_requirements"] == 0,
        "source384_receipt_bound_and_neural_replay_not_performed": (
            symbolic["source384"]["enabled"] is True and symbolic["source384"]["selected"] is True
            and symbolic["source384"]["inference_sha256"] is not None
            and symbolic["source384"]["neural_inference_replayed"] is False),
        "all_retries_disabled": retries_disabled,
        "supervisor_configs_equal_after_contract_output_identity_and_inactive_retry_order": normalize_config(symbolic_config, remove_contract=True) == normalize_config(direct_config, remove_contract=True),
        "strict_declared_common_controls_direct_vs_symbolic": selected[st["trial"]]["declared_controls_comparison"],
        "prepared_actual_selection_unchanged": boolean(symbolic_receipt.get("intent_selection_config_unchanged")),
        "official_result_bytes_bound": True,
        "actual_cleanup_quiescent": symbolic["remaining_processes"] == 0 and symbolic["cleanup_returncode"] == 0,
    }
    def subtract(left, right):
        return left - right if left is not None and right is not None else None
    def ratio(left, right):
        return left / right if left is not None and right not in (None, 0) else None
    dtokens, stokens = direct["tokens"]["total_tokens"], symbolic["tokens"]["total_tokens"]
    dseconds, sseconds = direct["durations_seconds"]["agent_execution"], symbolic["durations_seconds"]["agent_execution"]
    report = {
        "schema": "largest-eigenval-symbolic-single-trial-comparison@1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "canonical_receipt": False,
        "producer_sha256": sha(read(Path(__file__).resolve(), limit=128 * 1024)[0]),
        "reporting_runner_head": args.runner_head, "source_owners": source_owners,
        "archive_sha256": EXPECTED_ARCHIVE, "archive_bytes_rehashed_here": False,
        "task": "largest-eigenval", "rows": {"symbolic": symbolic, "historical_direct": direct, "historical_native": native},
        "checks": checks, "signed_native_admission_observations": admissions,
        "official_results": original_results,
        "candidate_contract": {"schema": contract["schema"], "canonical_sha256": sha(canonical(contract)),
            "raw_file_sha256": next(row["sha256"] for row in evidence if row.get("path") == str(args.contract.absolute())),
            "requirements": requirement_count, "ledger_sha256": digest(ledger.get("ledger_sha256")),
            "operations": len(operation_contract.get("operations", [])),
            "interpretation_scope": closed(operation_contract.get("interpretation_scope"), {"administrative_requirement_task_coverage"}),
            "source_accounting_complete": boolean(ledger.get("source_accounting_complete")),
            "semantic_support_complete": boolean(ledger.get("semantic_support_complete")),
            "semantic_alignment_verified": boolean(ledger.get("semantic_alignment_verified")),
            "proof_authority": boolean(ledger.get("proof_authority")),
            "execution_authority": boolean(ledger.get("execution_authority")),
            "completion_authority": boolean(ledger.get("completion_authority")),
            "source_semantics_verified": False, "authored_candidate_is_gold_semantic_translation": False,
            "numeric_correctness_established_by_plan": False, "benchmark_excluded_from_training": True},
        "observed_difference_symbolic_minus_historical_direct": {
            "router_sessions": subtract(symbolic["router_sessions"], direct["router_sessions"]),
            "total_tokens": subtract(stokens, dtokens), "total_tokens_ratio": ratio(stokens, dtokens),
            "agent_seconds": subtract(sseconds, dseconds), "agent_seconds_ratio": ratio(sseconds, dseconds),
            "planning_seconds": subtract(symbolic["supervisor_phase_seconds"]["planning_seconds"], direct["supervisor_phase_seconds"]["planning_seconds"]),
            "official_rewards_equal": symbolic["official_reward"] == direct["official_reward"] if symbolic["official_reward"] is not None else None,
            "descriptive_single_trial_only": True},
        "runtime_observation_binding": runtime_observation_binding,
        "runtime_observation": {
            "container_id": digest(runtime_observation.get("container_id")),
            "container_image_sha256": digest(str(runtime_observation.get("image_id", "")).removeprefix("sha256:")),
            "nano_cpus": number(obj(runtime_observation.get("actual_resources")).get("NanoCpus"), integer=True),
            "memory_bytes": number(obj(runtime_observation.get("actual_resources")).get("Memory"), integer=True),
            "memory_plus_swap_bytes": number(obj(runtime_observation.get("actual_resources")).get("MemorySwap"), integer=True),
        } if runtime_observation else None,
        "runtime_container_image_bytes_identical": False,
        "provider_call_count_basis": "native CLI sessions; supervisor llm_router invocations; not individual internal API or model turns",
        "cache_included_in_input_tokens": True, "unknown_usage_fields_are_zero": False,
        "billing_total_verified": False, "dollar_cost": None,
        "benchmark_advantage_claimed": False, "causal_efficiency_claim": False,
        "raw_model_verifier_or_solution_bodies_exported": False, "credential_contents_read": False,
        "completion_authority": False, "evidence": evidence,
        "limits": [
            "One fresh symbolic trial and two historical trials; no randomized, repeated, or causal efficiency estimate.",
            "Exact supervisor source archive is reused; historical task/container image fingerprints, time, host load, packages, provider state and session caches are not controlled.",
            "Only the reviewed administrative contract is the intended supervisor treatment; full config hashes retain output identity and inactive retry-list serialization differences.",
            "Administrative requirement coverage and a consistent plan do not prove source interpretation or floating-point/complex eigensolver behavior.",
            "Official benchmark correctness and speed are empirical finite checks, separate from native task completion and usage completeness.",
            "CLI/router sessions can contain many internal model turns; token-count event counts are observations, not provider-call counts.",
            "Unknown or interrupted complete usage remains null while observed failed-call counters remain known subtotals.",
            "Cached input tokens are already included in input; unverified provider dollar amounts and Harbor price estimates are excluded.",
            "Cleanup return code and remaining process count are observed; the frozen runtime does not report a separate cleanup duration.",
            "Signed admission bytes are bound here; this metadata producer does not replay cryptographic admission or rerun any verifier.",
            "Source384 neural_inference_replayed=False is the consumer's no-replay invariant, not evidence that initial inference was disabled; original inference worker bytes are not separately reexecuted here.",
        ],
    }
    # Check that source-bound receipt bytes stayed unchanged through collect().
    for binding in evidence:
        if "git_head" not in binding and read(Path(binding["path"]))[1]["sha256"] != binding["sha256"]:
            raise ValueError("metadata input changed before publication")
    encoded = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) > 256 * 1024:
        raise ValueError("bounded comparison exceeded output allowance")
    with args.output.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps({"output": str(args.output), "sha256": sha(encoded),
        "symbolic_reward": symbolic["official_reward"], "symbolic_sessions": symbolic["router_sessions"],
        "symbolic_tokens": symbolic["tokens"], "historical_direct_sessions": direct["router_sessions"],
        "symbolic_planning_zero_provider_calls": checks["actual_symbolic_planning_zero_provider_calls"],
        "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
