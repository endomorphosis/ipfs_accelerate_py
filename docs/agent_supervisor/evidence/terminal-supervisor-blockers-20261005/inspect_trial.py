"""One-shot, bounded metadata projection; never follows model/verifier paths.

Usage: python inspect_trial.py --trial largest-eigenval-02 \
  --build-command build-02-command.json --output trial-02-summary.json
Paths may be absolute or relative to this script. Existing outputs are refused.
The frozen measured_usage function is reused from the build's exact Git source,
after verifying its entire source file against the runtime archive manifest.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 16 * 1024 * 1024
OWNER = "benchmarks/agent_supervisor/container_coding/full_supervisor_harbor_agent.py"
CODEC = "ipfs_accelerate_py/agent_supervisor/runtime/semantic_router_translation.py"
FAILURE_PHASES = {"provider_invocation", "provider_result_validation", "semantic_response_decode"}
PROVIDER_REASONS = {"authentication", "billing", "rate_limit", "quota", "invalid_request",
                    "not_found", "timeout", "server", "transport", "policy", "unknown"}
STATUSES = {"succeeded", "failed", "denied", "conflict", "not_found", "cancelled", "timed_out",
            "unavailable", "completed", "blocked", "pending", "ready", "in_progress", "retrying",
            "stopped", "running", "residual", "repaired", "abstained", "available", "unsupported"}
ENUM_OWNERS = {
    "driver": "benchmarks/agent_supervisor/container_coding/terminal_container_supervisor.py",
    "progress": "benchmarks/agent_supervisor/container_coding/terminal_native_progress.py",
    "doctor_dispatch": "benchmarks/agent_supervisor/container_coding/terminal_doctor_dispatch.py",
    "doctor_workflow": "ipfs_accelerate_py/agent_supervisor/runtime/doctor_task_workflow.py",
    "cache": "benchmarks/agent_supervisor/container_coding/terminal_setup_cache_advice.py",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def read(p, limit=MAX_BYTES):
    p = Path(p)
    if p.resolve(strict=True) != p:
        raise ValueError("noncanonical inspector input")
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("inspector input must be a bounded regular file")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > limit or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("inspector input changed")
    return raw


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate inspector input key")
        result[key] = value
    return result


def decode(raw):
    return json.loads(raw, object_pairs_hook=strict_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite inspector input")))


def obj(value):
    return value if type(value) is dict else {}


def number(value, *, integer=False):
    accepted = type(value) is int if integer else type(value) in (int, float)
    return value if accepted and math.isfinite(value) and 0 <= value <= 2**63 - 1 else None


def boolean(value):
    return value if type(value) is bool else None


def closed(value, allowed):
    return None if value is None else value if type(value) is str and value in allowed else "unrecognized"


def identity(value):
    return value if type(value) is str and re.fullmatch(r"[A-Za-z0-9_.@+-]{1,128}", value) else None


def digest(value):
    return value if type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) else None


def error_type(value):
    return value if type(value) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,79}", value) else None


def counts(value, fields):
    value = obj(value)
    return {key: number(value.get(key), integer=True) for key in fields}


def times(value, fields):
    value = obj(value)
    return {key: number(value.get(key)) for key in fields}


def command_arg(recipe, flag):
    argv = recipe.get("argv")
    if type(argv) is not list or argv.count(flag) != 1:
        raise ValueError("build command argument missing or ambiguous")
    value = argv[argv.index(flag) + 1]
    if type(value) is not str:
        raise ValueError("build command argument must be text")
    return value


def frozen_source(recipe, manifest, relative):
    source = command_arg(recipe, "--source")
    head = obj(recipe.get("source_heads")).get(source)
    if type(head) is not str or not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("exact source head required")
    raw = subprocess.check_output(["git", "-C", source, "show", head + ":" + relative], timeout=30)
    rows = [row for row in manifest.get("files", [])
            if type(row) is dict and row.get("path") == "source/" + relative]
    if len(raw) > 512 * 1024 or len(rows) != 1 or rows[0].get("sha256") != sha(raw):
        raise ValueError("inspector owner differs from frozen runtime source")
    return raw


def usage_owner(recipe, manifest):
    raw = frozen_source(recipe, manifest, OWNER)
    tree = ast.parse(raw)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "measured_usage"]
    if len(selected) != 1:
        raise ValueError("one frozen measured_usage owner required")
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), OWNER, "exec"), namespace)
    return namespace["measured_usage"], sha(raw)


def semantic_codes(recipe, manifest):
    raw = frozen_source(recipe, manifest, CODEC)
    assignments = [node for node in ast.parse(raw).body if isinstance(node, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "_ERROR_REASON_CODES" for t in node.targets)]
    mapping = ast.literal_eval(assignments[0].value) if len(assignments) == 1 else {}
    if type(mapping) is not dict or len(mapping) > 128 or any(identity(v) is None for v in mapping.values()):
        raise ValueError("closed semantic reason code mapping required")
    return set(mapping.values()) | {"unclassified"}, sha(raw)


def native_enums(recipe, manifest):
    """Derive closed labels from exact archived owners, never receipt strings."""
    raws = {name: frozen_source(recipe, manifest, relative) for name, relative in ENUM_OWNERS.items()}
    trees = {name: ast.parse(raw) for name, raw in raws.items()}

    def assigned_strings(tree, name):
        return {node.value.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            and isinstance(node.value, ast.Constant) and type(node.value.value) is str}

    def fields(tree, key):
        values = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                values.update(v.value for k, v in zip(node.keys, node.values)
                    if isinstance(k, ast.Constant) and k.value == key
                    and isinstance(v, ast.Constant) and type(v.value) is str)
            elif isinstance(node, ast.keyword) and node.arg == key:
                if isinstance(node.value, ast.Constant) and type(node.value.value) is str:
                    values.add(node.value.value)
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and type(node.value.value) is str:
                if any(isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                       and t.slice.value == key for t in node.targets):
                    values.add(node.value.value)
        return values

    def declared_set(tree, name):
        nodes = [node.value for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
        if (len(nodes) != 1 or not isinstance(nodes[0], ast.Call)
                or not isinstance(nodes[0].func, ast.Name) or nodes[0].func.id != "frozenset"
                or len(nodes[0].args) != 1):
            raise ValueError("closed native enum declaration required")
        value = ast.literal_eval(nodes[0].args[0])
        if type(value) is not set or len(value) > 128 or any(type(v) is not str for v in value):
            raise ValueError("bounded native string enum required")
        return value

    progress = trees["progress"]
    terminal = declared_set(progress, "TERMINAL_STATUSES")
    fixed_stops = {"expired_attempt_settlement_unavailable",
        "timeout_observed_without_terminal_state", "run_ended_without_terminal_observation"}
    literals = {node.value for node in ast.walk(progress)
                if isinstance(node, ast.Constant) and type(node.value) is str}
    if not fixed_stops <= literals:
        raise ValueError("native progress stop-owner grammar changed")
    enums = {
        "phases": assigned_strings(trees["driver"], "phase"),
        "stop_reasons": {"native_task_" + value for value in terminal}
            | declared_set(progress, "QUIESCENT_STOP_REASONS") | fixed_stops,
        "idle_reasons": declared_set(progress, "IDLE_REASONS"),
        "doctor_routes": fields(trees["doctor_dispatch"], "route"),
        "doctor_statuses": fields(trees["doctor_dispatch"], "status") | fields(trees["doctor_workflow"], "status"),
        "doctor_analysis_statuses": fields(trees["doctor_dispatch"], "analysis_status") | fields(trees["doctor_workflow"], "analysis_status"),
        "cache_phases": fields(trees["cache"], "phase"),
    }
    if any(not values or len(values) > 128 or any(type(v) is not str or
            (v != "" and identity(v) is None) for v in values) for values in enums.values()):
        raise ValueError("native owner enum must remain bounded and closed")
    return enums, {name: sha(raw) for name, raw in raws.items()}


def project_invocation(row, semantic_reasons):
    row = obj(row)
    usage = obj(row.get("usage"))
    result = {"phase": closed(row.get("phase", row.get("purpose")), {"planning", "coding"}),
        "status": closed(row.get("status"), {"provider_returned", "failed"}),
        "provider": closed(row.get("provider"), {"codex_cli", "grok_cli"}),
        "model": identity(row.get("model")), "seconds": number(row.get("seconds")),
        "error_type": error_type(row.get("error_type")),
        "failure_phase": closed(row.get("failure_phase"), FAILURE_PHASES),
        "exit_code": usage.get("exit_code") if type(usage.get("exit_code")) is int and -255 <= usage["exit_code"] <= 255 else None,
        "timed_out": boolean(usage.get("timed_out")),
        "native_token_count_records": number(obj(row.get("native_rollout_usage")).get("observed_token_count_records"), integer=True)}
    for field, allowed in (("provider_failure", PROVIDER_REASONS), ("semantic_response_failure", semantic_reasons)):
        value = obj(row.get(field))
        result[field] = None if not value else {"phase": closed(value.get("phase"), FAILURE_PHASES),
            "reason_code": closed(value.get("reason_code"), allowed)}
        status = value.get("http_status")
        if field == "provider_failure" and type(status) is int and 100 <= status <= 599:
            result[field]["http_status"] = status
    return result


def project_trial(row, measured_usage, semantic_reasons, enums):
    supervisor = obj(row.get("supervisor"))
    initial = obj(supervisor.get("initial_context"))
    source = obj(initial.get("source384_context"))
    progress = obj(supervisor.get("native_progress"))
    latest = obj(progress.get("latest"))
    latest_task = obj(latest.get("task"))
    latest_heartbeat = obj(latest.get("heartbeat"))
    context = obj(supervisor.get("context"))
    doctor = obj(supervisor.get("doctor_dispatch"))
    planning = obj(supervisor.get("planning"))
    metadata = obj(obj(row.get("agent_context")).get("metadata"))
    cache = obj(metadata.get("setup_cache"))
    invocations = supervisor.get("provider_invocations", [])
    if type(invocations) is not list or len(invocations) > 64 or any(type(v) is not dict for v in invocations):
        raise ValueError("bounded provider invocation list required")
    usage = measured_usage(supervisor)
    # The existing owner preserves unknown totals as null, including failed
    # calls with no token receipts. Cached input is already included in input.
    return {
        "trial": identity(row.get("trial")),
        "exact_trial_task_matches": boolean(row.get("exact_trial_task_matches")),
        "official_reward": number(obj(row.get("reward")).get("reward")),
        "exception_type": error_type(row.get("exception_type")),
        "durations_seconds": times(row.get("durations_seconds"), ("environment_setup", "agent_setup", "agent_execution", "verifier")),
        "usage": usage,
        "provider_invocations": [project_invocation(v, semantic_reasons) for v in invocations],
        "supervisor": {"task_completed": boolean(supervisor.get("task_completed")),
            "seconds": number(supervisor.get("seconds")),
            "error_type": error_type(obj(supervisor.get("error")).get("type")),
            "error_phase": closed(supervisor.get("error_phase"), enums["phases"]),
            "remaining_processes": number(supervisor.get("remaining_processes"), integer=True),
            "worker_cleanup_returncode": number(supervisor.get("worker_cleanup_returncode"), integer=True),
            "phases": times(supervisor.get("phases"), tuple(x + "_seconds" for x in
                ("prepare", "initial_context", "planning", "context", "doctor", "post_publication_refresh"))),
            "start_status": closed(obj(supervisor.get("start")).get("status"), STATUSES),
            "stop_status": closed(obj(supervisor.get("stop")).get("status"), STATUSES),
            "task_status": closed(obj(supervisor.get("task_state")).get("status"), STATUSES)},
        "planning": {**counts(planning, ("goals", "tasks", "provider_calls")),
            "qualified": boolean(planning.get("qualified")), "elapsed_seconds": number(planning.get("elapsed_seconds")),
            "strategy": closed(planning.get("planning_strategy"), {"direct", "symbolic"})},
        "initial_context": {**counts(initial, ("indexed_symbols", "full_capsules", "world_task_count", "provider_calls")),
            "seconds": number(initial.get("seconds")), "canonical_tasks_created": boolean(initial.get("canonical_tasks_created")),
            "learned_embeddings": boolean(initial.get("learned_embeddings"))},
        "admitted_context": counts(context, ("indexed_symbols", "full_capsules", "worker_capsules", "world_task_count", "new_embedding_calls")),
        "source384": {"checkpoint_sha256": digest(source.get("checkpoint_sha256")),
            "inference_sha256": digest(source.get("inference_sha256")), "config_sha256": digest(source.get("config_sha256")),
            "seconds": number(source.get("seconds")), "training_steps": number(source.get("training_steps"), integer=True),
            "neural_inference_replayed": boolean(source.get("neural_inference_replayed")),
            "counts": counts(source.get("summary"), ("source_files", "program_source_files", "harness_support_files", "inference_python_files", "omitted_candidates", "provider_calls"))},
        "doctor": {"status": closed(doctor.get("status"), enums["doctor_statuses"]),
            "route": closed(doctor.get("route"), enums["doctor_routes"]),
            "analysis_status": closed(doctor.get("analysis_status"), enums["doctor_analysis_statuses"]), "provider_calls": number(doctor.get("provider_calls"), integer=True)},
        "native_progress": {**counts(progress, ("samples", "transition_count", "transitions_omitted", "provider_outcomes_omitted")),
            "stop_reason": closed(progress.get("stop_reason"), enums["stop_reasons"]),
            "latest": {"seconds": number(latest.get("seconds")),
                "task_status": closed(latest_task.get("status"), STATUSES),
                "task_revision": number(latest_task.get("revision"), integer=True),
                "selection_idle_reason": closed(latest_heartbeat.get("selection_idle_reason"), enums["idle_reasons"]),
                "heartbeat_sequence": number(latest_heartbeat.get("sequence"), integer=True),
                "active_task_present": boolean(latest_heartbeat.get("active_task_present")),
                "claimed_task_present": boolean(latest_heartbeat.get("claimed_task_present"))}},
        "setup_cache": {"policy": closed(cache.get("policy"), {"source384-native-aarch64-dontneed@1", "source384-native-aarch64-dontneed@2"}),
            "completed": boolean(cache.get("completed")),
            "phase": closed(cache.get("phase"), enums["cache_phases"])},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--build-command", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = path(args.output)
    if output.exists():
        raise ValueError("fresh inspection output required")
    trial = path(args.trial)
    receipt_raw = read(trial / "receipt.json")
    receipt = decode(receipt_raw)
    prepared_raw = read(trial / "preparation.json")
    prepared = decode(prepared_raw)
    build_raw = read(path(args.build_command), 1024 * 1024)
    build = decode(build_raw)
    archive = Path(prepared["archive"])
    if archive != Path(command_arg(build, "--output")):
        raise ValueError("trial archive differs from selected build")
    manifest_raw = read(archive / "manifest.json")
    manifest = decode(manifest_raw)
    if sha(manifest_raw) != prepared.get("manifest_sha256") or manifest.get("archive_sha256") != prepared.get("archive_sha256"):
        raise ValueError("frozen archive manifest binding differs")
    for name in ("model", "reasoning_effort", "cli_version"):
        if receipt.get(name) != prepared.get(name) or identity(receipt.get(name)) is None:
            raise ValueError("receipt provider identity differs from preparation")
    if manifest.get("codex_version") != receipt["cli_version"]:
        raise ValueError("frozen CLI differs from runtime manifest")
    measured_usage, owner_sha = usage_owner(build, manifest)
    semantic_reasons, codec_sha = semantic_codes(build, manifest)
    enums, enum_owner_sha256 = native_enums(build, manifest)
    rows = receipt.get("trials")
    if type(rows) is not list or len(rows) != 1 or receipt.get("trial_count") != 1:
        raise ValueError("exactly one completed trial receipt required")
    heads, clean = {}, {}
    for label, flag in (("accelerate", "--source"), ("datasets", "--datasets"), ("kit", "--kit")):
        root = command_arg(build, flag)
        head = obj(build.get("source_heads")).get(root)
        if type(head) is not str or not re.fullmatch(r"[0-9a-f]{40}", head):
            raise ValueError("exact build source heads required")
        heads[label] = head
        clean[label] = obj(build.get("source_status")).get(root) == ""
    result = {"schema": "terminal-supervisor-bounded-trial-summary@2", "canonical_receipt": False,
        "inspected_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": identity(receipt.get("task")), "arm": closed(receipt.get("arm"), {"full", "no-index"}),
        **{name: receipt[name] for name in ("model", "reasoning_effort", "cli_version")},
        "resource_profile": closed(receipt.get("resource_profile"), {"source384-5cpu-12gib@1", "source384-5cpu-16gib-extended@1"}),
        "source_heads": heads, "build_sources_clean": clean,
        "receipt_sha256": sha(receipt_raw), "receipt_bytes": len(receipt_raw),
        "build_command_sha256": sha(build_raw), "preparation_sha256": sha(prepared_raw),
        "manifest_sha256": sha(manifest_raw), "archive_sha256": digest(manifest.get("archive_sha256")),
        "archive_bytes_rehashed_here": False,
        "measured_usage_owner_sha256": owner_sha, "semantic_diagnostic_owner_sha256": codec_sha,
        "closed_enum_owner_sha256": enum_owner_sha256,
        "inspector_sha256": sha(read(Path(__file__).resolve(), 128 * 1024)),
        "harbor_returncode": number(receipt.get("harbor_returncode"), integer=True),
        "invocation_seconds": number(receipt.get("invocation_seconds")),
        "original_task_inputs_unchanged": boolean(receipt.get("original_task_inputs_unchanged")),
        "complete_single_trial_receipt": boolean(receipt.get("complete_single_trial_receipt")),
        "trials": [project_trial(rows[0], measured_usage, semantic_reasons, enums)],
        "exported_model_or_verifier_bodies": False, "exported_raw_error_strings": False,
        "credential_contents_read": False, "completion_authority": False, "benchmark_advantage_claimed": False}
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if len(encoded.encode()) > 256 * 1024:
        raise ValueError("bounded summary output exceeded")
    with output.open("x") as stream:
        stream.write(encoded)
    print(json.dumps({"output": output.name, "sha256": sha(encoded.encode()),
        "official_reward": result["trials"][0]["official_reward"], "usage": result["trials"][0]["usage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
