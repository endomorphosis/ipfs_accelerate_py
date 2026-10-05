"""Bounded native-versus-supervisor observation; never exports model bodies."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

TOKEN_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
PHASE_KEYS = ("environment_setup", "agent_setup", "agent_execution", "verifier")


def read(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("bounded regular input required")
    data = path.read_bytes()
    return json.loads(data), {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def number(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def counters(value):
    return {key: value.get(key) if type(value.get(key)) is int and value[key] >= 0 else None for key in TOKEN_KEYS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout))
    from benchmarks.agent_supervisor.container_coding.benchmark_comparison import collect
    b, bb = read(args.baseline)
    s, sb = read(args.supervisor)
    comparison = collect(baseline=args.baseline, supervisors=[args.supervisor])
    if len(comparison["rows"]) != 2 or b["trial_count"] != 1 or s["trial_count"] != 1:
        raise ValueError("one original trial per arm required")
    bt = b["trials"][0]
    st = s["trials"][0]
    observed_sessions = []
    for row in bt["raw_usage"]["sessions"]:
        observed_sessions.append({
            "session_id_sha256": hashlib.sha256(row["session_id"].encode()).hexdigest(),
            "rollout_sha256": row["sha256"],
            "cli_version": row["cli_version"],
            "observed_models": row["observed_models"],
            "observed_reasoning_efforts": row["observed_reasoning_efforts"],
            "task_complete_observed": row["task_complete_observed"],
            "token_count_events": row["token_count_events"],
            "malformed_lines": row["malformed_lines"],
            "usage": counters(row["usage"]),
        })
    trial_name = bt["trial"]
    if not isinstance(trial_name, str) or re.fullmatch(r"largest-eigenval__[A-Za-z0-9]+", trial_name) is None:
        raise ValueError("unexpected original task identity")
    result_path = args.baseline.parent / "jobs" / "native-codex-largest-eigenval" / trial_name / "result.json"
    result, rb = read(result_path)
    if rb["sha256"] != bt["result_sha256"]:
        raise ValueError("native result bytes differ from receipt")
    if (result.get("verifier_result") or {}).get("rewards") != bt["reward"]:
        raise ValueError("official reward differs from receipt")
    bc, bcb = read(args.baseline.parent / "config.json")
    sc, scb = read(args.supervisor.parent / "config.json")
    normalized_retry = lambda x: {k: sorted(v) if k in ("exclude_exceptions", "include_exceptions") and isinstance(v, list) else v for k, v in x.items()}
    current = []
    for row in comparison["rows"]:
        current.append({key: row[key] for key in (
            "arm", "trial", "task", "model", "reasoning_effort", "cli_version", "official_reward", "outcome", "exception_type",
            "durations_seconds", "provider_calls", "native_usage_complete_observed", "tokens", "known_token_subtotals",
            "reported_identity_matches", "declared_controls_comparison", "comparison_profile_matches", "original_task_inputs_unchanged")})
    native = next(row for row in current if row["arm"] == "native-codex")
    supervisor = next(row for row in current if row["arm"] == "full")
    nt, tt = native["tokens"]["total_tokens"], supervisor["tokens"]["total_tokens"]
    ng, tg = native["durations_seconds"]["agent_execution"], supervisor["durations_seconds"]["agent_execution"]
    all_runtime_identity_observed = bool(observed_sessions) and all(
        row["cli_version"] == b["cli_version"] and row["observed_models"] == [b["model"]]
        and row["observed_reasoning_efforts"] == [b["reasoning_effort"]]
        for row in observed_sessions)
    report = {
        "schema": "native-eigenvalue-single-trial-comparison@1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inspector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison_source_sha256": comparison["comparison_source_sha256"],
        "task": "largest-eigenval",
        "rows": current,
        "native_observed_sessions": observed_sessions,
        "checks": {
            "native_receipt_complete": b["complete_single_trial_receipt"],
            "native_job_returncode_zero": b.get("harbor_returncode") == 0,
            "native_original_result_hash_matches": True,
            "native_result_finished": bool(result.get("finished_at")),
            "native_runtime_cli_model_reasoning_observed": all_runtime_identity_observed,
            "native_declared_trial_profile_matches": bt["exact_trial_profile_matches"],
            "native_usage_corrobated_by_job": b["usage_corroborated_by_unscrubbed_native_job"],
            "both_original_task_inputs_unchanged": b["original_task_inputs_unchanged"] is True and s["original_task_inputs_unchanged"] is True,
            "both_retry_policies_disabled": bc["retry"]["max_retries"] == sc["retry"]["max_retries"] == 0,
            "retry_policies_semantically_equal": normalized_retry(bc["retry"]) == normalized_retry(sc["retry"]),
            "all_declared_controls_exactly_match": comparison["all_declared_controls_match"],
        },
        "observed_difference": {
            "supervisor_total_tokens_minus_native": tt - nt if nt is not None and tt is not None else None,
            "supervisor_tokens_divided_by_native": tt / nt if nt and tt is not None else None,
            "supervisor_agent_seconds_minus_native": tg - ng if ng is not None and tg is not None else None,
            "supervisor_agent_seconds_divided_by_native": tg / ng if ng and tg is not None else None,
            "descriptive_single_trial_only": True,
        },
        "budget_seconds": {"outer_agent_both": 960, "setup_both": 1800, "supervisor_work": 840, "supervisor_cleanup_reserve": 60, "supervisor_driver": 900, "supervisor_planning_call": 90, "supervisor_coding_call": 300, "native_session": 960},
        "observed_outer_invocation_seconds": {"native": number(b.get("invocation_seconds")), "supervisor": number(s.get("invocation_seconds"))},
        "cache_included_in_input_tokens": True,
        "provider_call_count_basis": "native sessions and supervisor router invocations; not individual internal model turns",
        "billing_total_verified": False,
        "dollar_cost": None,
        "benchmark_advantage_claimed": False,
        "causal_claim": False,
        "raw_model_or_verifier_bodies_exported": False,
        "limits": ["One retrospective trial per arm; not a randomized or repeated benchmark estimate.", "Same outer limits do not imply identical internal work/provider-call allowances or tool permissions.", "Setup work, indexed-context preparation and cache treatment differ between arms.", "Strict configuration hashing retains retry-list order differences, even with identical disabled retry policy.", "Unknown or interrupted usage remains unknown; cached input is already included in input."],
        "evidence": [bb, sb, rb, bcb, scb],
    }
    if args.output.exists():
        raise ValueError("fresh comparison output required")
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "native_reward": native["official_reward"], "native_tokens": native["tokens"], "native_agent_seconds": ng, "supervisor_reward": supervisor["official_reward"], "supervisor_tokens": supervisor["tokens"], "supervisor_agent_seconds": tg, "native_runtime_identity_observed": all_runtime_identity_observed}, sort_keys=True))


if __name__ == "__main__":
    main()
