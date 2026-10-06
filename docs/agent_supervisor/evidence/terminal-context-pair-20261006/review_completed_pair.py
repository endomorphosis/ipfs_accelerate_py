"""Review completed-pair metadata and report all retained campaign costs.

This producer must run only after final fresh receipts and comparison exist.
It does not invoke a model, benchmark, database, tokenizer or training job.
No transcript, log, task source, solution or hidden evaluator body is opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess

EXPECTED_SOURCE_HEAD = "82671e6c8411875cb9919e3a71b07831548c0aeb"
EXPECTED_COMPARATOR = "e3cde58b7e8c98962582c98adcbd988cc570f65b1e7dd75bbacf12387dcfd1ac"
TOKENS = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
MAX_BYTES = 64 * 1024 * 1024


def read(path, bindings, *, json_object=True):
    path = Path(path).absolute()
    before = path.lstat()
    if (path.resolve(strict=True) != path or not stat.S_ISREG(before.st_mode)
            or before.st_size > MAX_BYTES):
        raise ValueError("bounded canonical regular input required")
    raw = path.read_bytes()
    after = path.stat()
    if before.st_ino != after.st_ino or before.st_mtime_ns != after.st_mtime_ns or len(raw) != before.st_size:
        raise ValueError("input changed during review")
    binding = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    bindings.append(binding)
    if not json_object:
        return raw, binding
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate metadata field")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite metadata")))
    if type(value) is not dict:
        raise ValueError("metadata object required")
    return value, binding


def check(checks, name, condition):
    checks.append({"name": name, "passed": condition is True})


def is_count(value):
    return type(value) is int and value >= 0


def elapsed(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def totals(receipt):
    """Independently add each native cumulative session once."""
    trials = receipt.get("trials")
    if type(trials) is not list or len(trials) != 1 or receipt.get("trial_count") != 1:
        raise ValueError("exactly one retained native trial required")
    trial = trials[0]
    supervisor = trial.get("supervisor") or {}
    invocations = supervisor.get("provider_invocations") or []
    if type(invocations) is not list:
        raise ValueError("provider invocation list required")
    known = {key: [] for key in TOKENS}
    seen = set()
    complete = bool(invocations)
    for invocation in invocations:
        identity = invocation.get("invocation_id")
        if type(identity) is not str or not identity or identity in seen:
            raise ValueError("distinct native invocation identities required")
        seen.add(identity)
        native = invocation.get("native_rollout_usage") or {}
        usage = native.get("usage") or {}
        available = all(is_count(usage.get(key)) for key in TOKENS)
        valid = (available and usage["input_tokens"] + usage["output_tokens"] == usage["total_tokens"]
            and usage["cached_input_tokens"] <= usage["input_tokens"])
        complete = (complete and valid and native.get("usage_available") is True
            and native.get("task_complete_observed") is True and native.get("cache_included_in_input") is True)
        for key in TOKENS:
            if is_count(usage.get(key)):
                known[key].append(usage[key])
    metadata = (trial.get("agent_context") or {}).get("metadata") or {}
    summary = metadata.get("usage") or {}
    complete = (complete and summary.get("all_invocations_receipted") is True
        and summary.get("provider_calls") == len(seen) and not supervisor.get("unreceipted_provider_attempt"))
    subtotal = {key: sum(items) if items else None for key, items in known.items()}
    total = subtotal if complete else {key: None for key in TOKENS}
    return {"trial": trial["trial"], "native_sessions": len(seen),
        "phases": [item.get("phase") for item in invocations],
        "official_reward": (trial.get("reward") or {}).get("reward"),
        "complete_native_usage_observed": complete, "tokens": total, "known_token_subtotals": subtotal,
        "provider_session_results": [{key: invocation.get(key) for key in (
            "phase", "status", "error_type", "failure_phase")}
            | {"seconds": elapsed(invocation.get("seconds")),
               "observed_exit_code": (invocation.get("usage") or {}).get("exit_code"),
               "semantic_response_failure_reason": (invocation.get("semantic_response_failure") or {}).get("reason_code")}
            for invocation in invocations],
        "supervisor_seconds": elapsed(supervisor.get("seconds")),
        "supervisor_error_type": (supervisor.get("error") or {}).get("type"),
        "supervisor_error_phase": supervisor.get("error_phase"),
        "task_status": (supervisor.get("task_state") or {}).get("status"),
        "remaining_processes": supervisor.get("remaining_processes"),
        "worker_cleanup_returncode": supervisor.get("worker_cleanup_returncode")}


def sum_accounting(rows, key):
    values = [row["tokens"][key] for row in rows]
    return sum(values) if all(value is not None for value in values) else None


def markdown(payload):
    lines = ["# Retained planning and coding transport comparison", "",
        "Both fresh arms use the same frozen runtime and retain LLM planning and coding through the router.", "",
        "| Attempt | Official reward | Native sessions | Input | Cached input¹ | Output | Total |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, item in (("First baseline: planning failure", payload["earlier_failed_baseline"]),
                       ("Fresh baseline @1", payload["fresh_pair"][0]),
                       ("Fresh compact @2", payload["fresh_pair"][1])):
        values = [name, item["official_reward"], item["native_sessions"],
            *[item["tokens"][key] for key in TOKENS]]
        lines.append("| " + " | ".join("unknown" if value is None else str(value) for value in values) + " |")
    lines += ["", "¹ Cached input is already included in input tokens.", "",
        "The first baseline consumed 23,737 complete native tokens before strict plan admission rejected an extra task title. Coding never ran. Compact-01 was prepared but never executed; it is not counted as a completed zero-cost trial.", ""]
    pair = payload["pair_observation"]
    delta = pair["baseline_minus_compact_total_tokens"]
    if payload["comparison_qualified"] and delta is not None:
        direction = "fewer" if delta >= 0 else "more"
        lines.append(f"The compact run used {abs(delta):,} {direction} total tokens in this pair.")
    else:
        lines.append("The pair did not satisfy every comparison gate; its retained costs remain visible without a qualified saving claim.")
        if delta is not None:
            lines += ["", "Observed baseline minus compact total tokens: " + f"{delta:,}" + "."]
    compact = payload["fresh_pair"][1]
    coding = [item for item in compact["provider_session_results"] if item["phase"] == "coding"]
    if len(coding) == 1 and coding[0]["failure_phase"] == "semantic_response_decode":
        outcome = coding[0]
        lines += ["", "The compact coding CLI exited " + str(outcome["observed_exit_code"])
            + " after a " + f"{outcome['seconds']:.2f}" + "-second session. The router then rejected its structured response with `"
            + str(outcome["semantic_response_failure_reason"]) + "`. This is a strict response-envelope rejection, not a provider timeout."]
        if compact["supervisor_error_type"] == "TimeoutError":
            lines += ["", "The native task remained `" + str(compact["task_status"])
                + "`; the supervisor later exhausted its agent budget at " + f"{compact['supervisor_seconds']:.2f}"
                + " seconds. Cleanup recorded " + str(compact["remaining_processes"])
                + " remaining processes and worker cleanup return code " + str(compact["worker_cleanup_returncode"])
                + ". The later lifecycle timeout does not make the already completed CLI session's token counts incomplete."]
    lines += ["", "One pair cannot establish a reliable or causal saving percentage. Internal model turns, solution choices, tool use, cache occupancy and output length can differ.", "",
        "Prompt byte measurements cover complete initial task input, including appended advisories, and exclude provider system context and later internal history. Native cumulative token counters include the observed complete sessions.", "",
        "The campaign total includes the failed first baseline and both fresh arms. Observed complete campaign tokens: " + str(payload["campaign_accounting"]["total_tokens"]) + ".", "",
        "This review reads bounded metadata and implementation digests only. No model transcript, thought, task source, solution or hidden evaluator body is opened.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--earlier-failed-baseline", type=Path, required=True)
    parser.add_argument("--unexecuted-compact", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if any(path.exists() or path.absolute().resolve() != path.absolute() for path in (args.output, args.report)):
        raise ValueError("fresh canonical review/report outputs required")
    bindings, checks = [], []
    pair, _ = read(args.comparison, bindings)
    check(checks, "comparison_schema", pair.get("schema") == "terminal-planning-retained-transport-pair-accounting@1")
    check(checks, "frozen_comparator_unchanged", pair.get("producer_sha256") == EXPECTED_COMPARATOR)
    source = args.source_checkout.absolute()
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    check(checks, "frozen_runtime_head_and_clean_checkout", head == EXPECTED_SOURCE_HEAD and status == "")
    check(checks, "fresh_pair_arm_count", type(pair.get("arms")) is list and len(pair["arms"]) == 2)
    if not checks[-1]["passed"]:
        raise ValueError("two retained comparison arms required")
    fresh = []
    for index, directory in enumerate((args.baseline, args.compact)):
        receipt, receipt_binding = read(directory / "receipt.json", bindings)
        observed = totals(receipt)
        fresh.append(observed)
        arm = pair["arms"][index]
        row = arm["row"]
        label = "baseline" if index == 0 else "compact"
        check(checks, label + "_receipt_bound", any(item.get("path") == receipt_binding["path"]
            and item.get("sha256") == receipt_binding["sha256"] for item in arm["inputs"]))
        check(checks, label + "_independent_native_token_sum", row["tokens"] == observed["tokens"]
            and row["known_token_subtotals"] == observed["known_token_subtotals"])
        check(checks, label + "_trial_and_reward_preserved", row["trial"] == observed["trial"]
            and row["official_reward"] == observed["official_reward"])
        check(checks, label + "_planning_and_coding_retained", observed["native_sessions"] == 2
            and sorted(observed["phases"]) == ["coding", "planning"])
        check(checks, label + "_schema_custody_required", arm.get("planning_generation_schema_required") is True)
        supervisor = receipt["trials"][0].get("supervisor") or {}
        invocations = supervisor.get("provider_invocations") or []
        planning = [item for item in invocations if item.get("phase") == "planning"]
        coding = [item for item in invocations if item.get("phase") == "coding"]
        check(checks, label + "_direct_planning_and_codex_router", receipt.get("planning_strategy") == "direct"
            and (supervisor.get("planning") or {}).get("planning_strategy") == "direct"
            and all(item.get("provider") == "codex_cli" and item.get("router_calls") == 1 for item in invocations))
        if len(planning) == 1:
            policy = (planning[0].get("provider_invocation_policy") or {}).get("structured_output") or {}
            projection = policy.get("native_schema_projection") or {}
            usage = planning[0].get("usage") or {}
            check(checks, label + "_trusted_schema_observation_matches_projection",
                policy.get("schema") == "codex-native-planning-json-schema@1"
                and policy.get("native_schema_requested") is True and policy.get("response_schema_validated") is True
                and policy.get("plan_admitted") is False
                and projection.get("canonical_validation_preserved") is True
                and projection.get("shape_only") is True
                and usage.get("codex_output_schema_sha256") == projection.get("native_wire_schema_sha256")
                and usage.get("codex_output_schema_bytes") == projection.get("native_wire_schema_bytes"))
        if len(coding) == 1:
            invocation = coding[0]
            version = str(index + 1)
            translation = invocation.get("semantic_translation") or {}
            check(checks, label + "_transport_version_and_current_native_input",
                translation.get("schema") == "supervisor-semantic-router-encoding@" + version
                and translation.get("transport_schema", "supervisor-semantic-router-input@1")
                    == "supervisor-semantic-router-input@" + version
                and translation.get("freshness_checked") is True
                and translation.get("native_prompt_sha256") == invocation.get("native_prompt_sha256"))
            audit = supervisor.get("context_input_audit") or {}
            matches = [item for item in audit.get("matches", []) if item.get("invocation_id") == invocation.get("invocation_id")]
            check(checks, label + "_complete_native_and_model_input_audit",
                audit.get("all_observed_coding_inputs_verified") is True
                and audit.get("any_native_input_verified") is True and audit.get("any_model_input_verified") is True
                and any(item.get("model_input_verified") is True
                    and item.get("native_prompt_sha256") == invocation.get("native_prompt_sha256")
                    and item.get("model_prompt_sha256") == invocation.get("model_prompt_sha256")
                    and item.get("model_prompt_bytes") == invocation.get("model_prompt_bytes") for item in matches))
        for binding in arm["inputs"]:
            # Input lists contain only canonical metadata already named by the comparator.
            _, current = read(binding["path"], bindings)
            check(checks, label + "_input_digest_" + Path(binding["path"]).name,
                current["sha256"] == binding["sha256"] and current["bytes"] == binding["bytes"])
    for binding in pair.get("reused_comparison_sources", []) + pair.get("planning_generation_source_bindings", []):
        source_path = Path(binding["path"])
        source_path.relative_to(source)
        _, current = read(source_path, bindings, json_object=False)
        check(checks, "frozen_implementation_digest_" + source_path.name,
            current["sha256"] == binding["sha256"] and current["bytes"] == binding["bytes"])
    earlier_receipt, _ = read(args.earlier_failed_baseline / "receipt.json", bindings)
    earlier = totals(earlier_receipt)
    check(checks, "earlier_failed_baseline_cost_retained", earlier["tokens"]["total_tokens"] == 23737
        and earlier["official_reward"] == 0.0 and earlier["native_sessions"] == 1 and earlier["phases"] == ["planning"])
    prepared, _ = read(args.unexecuted_compact / "preparation.json", bindings)
    check(checks, "earlier_compact_prepared_but_unexecuted", prepared.get("prepared") is True
        and prepared.get("provider_calls") == 0 and not (args.unexecuted_compact / "invocation.json").exists()
        and not (args.unexecuted_compact / "receipt.json").exists())
    comparison = pair["comparison"]
    all_gates = all(item.get("passed") is True for arm in pair["arms"] for item in arm["qualification_checks"])
    all_gates = all_gates and all(item.get("passed") is True for item in comparison["qualification_checks"])
    check(checks, "qualified_flag_matches_all_gates", comparison.get("matched_pair_qualified") is all_gates)
    delta = fresh[0]["tokens"]["total_tokens"] - fresh[1]["tokens"]["total_tokens"] if all(
        item["tokens"]["total_tokens"] is not None for item in fresh) else None
    check(checks, "reported_total_difference_matches_native_counters",
        comparison["observed_baseline_minus_compact_tokens"]["total_tokens"] == delta)
    check(checks, "no_causal_or_benchmark_advantage_claim", comparison.get("repeatable_or_causal_reduction_established") is False
        and pair.get("benchmark_advantage_claimed") is False and pair.get("provider_cost_calculated") is False)
    rows = [earlier, *fresh]
    campaign = {key: sum_accounting(rows, key) for key in TOKENS}
    campaign["known_token_subtotals"] = {key: sum(item["known_token_subtotals"][key]
        for item in rows if item["known_token_subtotals"][key] is not None) for key in TOKENS}
    payload = {"schema": "terminal-planning-retained-completed-pair-review@1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "review_producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_head": head, "input_bindings": bindings, "checks": checks,
        "review_passed": all(item["passed"] for item in checks), "comparison_qualified": all_gates,
        "earlier_failed_baseline": earlier, "fresh_pair": fresh,
        "unexecuted_compact": {"path": str(args.unexecuted_compact.absolute()),
            "status": "prepared_not_executed", "completed_trial": False, "complete_tokens": None},
        "pair_observation": {"baseline_minus_compact_total_tokens": delta,
            "repeatable_or_causal_reduction_established": False}, "campaign_accounting": campaign,
        "new_provider_calls_by_review": 0, "source_or_runtime_edits_by_review": 0,
        "raw_model_data_read": False, "cached_input_already_in_input_total": True,
        "unknown_fields_are_zero": False}
    with args.output.open("x") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    with args.report.open("x") as stream:
        stream.write(markdown(payload))
    print(json.dumps({"review_passed": payload["review_passed"], "comparison_qualified": all_gates,
        "baseline_minus_compact_total_tokens": delta, "campaign_total_tokens": campaign["total_tokens"]}))
    if not payload["review_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
