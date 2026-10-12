"""Offline labeled comparison of retained full-supervisor receipts.

This does not invoke providers, read transcripts, synthesize a native baseline,
or merge distinct treatments into the collector's common ``full`` arm.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("bounded regular receipt required")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite JSON value")
    return json.loads(path.read_text(), object_pairs_hook=unique, parse_constant=nonfinite)


def normalized(config):
    result = copy.deepcopy(config)
    result.pop("jobs_dir")  # Each retained trial needs its own output directory.
    agents = result["agents"]
    if len(agents) != 1 or agents[0]["kwargs"]["arm"] != "full":
        raise ValueError("one full-supervisor agent required")
    agents[0]["kwargs"].pop("semantic_metadata_view", None)
    return result


def integer(value):
    return value if type(value) is int and value >= 0 else None


def extra_usage(report, rows):
    originals = {row["invocation_id"]: row for row in report.get("provider_invocations", [])}
    result = []
    for row in rows:
        raw = originals[row["invocation_id"]]
        native = raw.get("native_rollout_usage") or {}
        usage = native.get("usage") or {}
        complete = row["native_usage_complete_observed"] is True
        result.append({
            "invocation_id": row["invocation_id"], "phase": row["phase"],
            "reasoning_output_tokens": integer(usage.get("reasoning_output_tokens")) if complete else None,
            "known_reasoning_output_tokens": integer(usage.get("reasoning_output_tokens")),
            "cache_write_input_tokens": integer(usage.get("cache_write_input_tokens")) if complete else None,
            "known_cache_write_input_tokens": integer(usage.get("cache_write_input_tokens")),
            "reasoning_already_in_output": raw.get("provider") == "codex_cli",
            "cache_already_in_input": row["cache_included_in_input"],
            "extra_categories_added_to_total": False,
        })
    return result


def status_only(value):
    """Keep scalar status evidence; exclude generated plans and input bodies."""
    if not isinstance(value, dict):
        return None
    allowed = {"schema", "status", "reason", "enabled", "attempted", "fail_open",
               "checkpoint_sha256", "consumed_at_inference", "source384_enabled"}
    return {key: item for key, item in value.items() if key in allowed
            and (item is None or type(item) in (bool, int)
                 or isinstance(item, str) and len(item) <= 256)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, action="append", required=True)
    parser.add_argument("--treatment", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve(strict=True)))
    from benchmarks.agent_supervisor.container_coding import benchmark_comparison as comparison
    from benchmarks.agent_supervisor.container_coding import native_usage_accounting
    from benchmarks.agent_supervisor.container_coding.benchmark_controls import build_controls, compare_controls

    outputs, controls, reference = [], [], None
    baseline_controls = None
    identities = set()
    expected_modes = {"full-lexical-baseline": "legacy", "full-instruction-dedup": "common-bindings-instruction@1"}
    for label, paths in (("full-lexical-baseline", args.baseline), ("full-instruction-dedup", args.treatment)):
        for path in paths:
            path = path.resolve(strict=True)
            receipt, prepared, config = (read(path), read(path.parent / "preparation.json"),
                                         read(path.parent / "config.json"))
            if receipt.get("schema") != comparison.SUPERVISOR or receipt.get("arm") != "full":
                raise ValueError("native full-supervisor receipt required")
            configured_mode = config["agents"][0]["kwargs"].get("semantic_metadata_view", "legacy")
            if configured_mode != expected_modes[label]:
                raise ValueError("treatment label conflicts with immutable configuration")
            checks = {
                "prepared_config_unchanged": prepared["config_sha256"] == digest(path.parent / "config.json"),
                "original_task_inputs_unchanged": receipt.get("original_task_inputs_unchanged") is True,
                "verifier_enabled": config["verifier"]["disable"] is False,
                "complete_single_trial_receipt": receipt.get("complete_single_trial_receipt") is True,
            }
            rebuilt = build_controls(config, task_input_sha256=prepared["task_input_sha256"],
                                     **{key: prepared[key] for key in ("task", "model", "reasoning_effort", "cli_version")})
            checks["prepared_controls_recomputed"] = rebuilt == prepared.get("comparison_controls")
            checks["receipt_controls_bound_to_preparation"] = (
                (receipt.get("comparison_controls") or {}).get("declared") == rebuilt)
            checks["receipt_identity_bound_to_preparation"] = all(
                receipt.get(key) == prepared.get(key) for key in ("task", "model", "reasoning_effort", "cli_version"))
            if reference is None:
                reference = normalized(config)
                baseline_controls = receipt.get("comparison_controls")
            checks["all_other_config_controls_equal"] = normalized(config) == reference
            generic = compare_controls(baseline_controls, receipt.get("comparison_controls"))
            checks["common_declared_controls_match"] = generic["matches"] is True
            controls.append({"label": label, "receipt_sha256": digest(path), "checks": checks,
                             "common_control_comparison": generic})
            trials = receipt.get("trials")
            if not isinstance(trials, list) or len(trials) != receipt.get("trial_count"):
                raise ValueError("inconsistent native trial inventory")
            source = {"path": str(path), "sha256": digest(path), "trial_count": len(trials),
                      "invocation_seconds": comparison._number(receipt.get("invocation_seconds"))}
            rows = []
            for trial in trials:
                identity = receipt["arm"], receipt["task"], trial.get("trial")
                if identity in identities:
                    raise ValueError("duplicate or conflicting retained trial")
                identities.add(identity)
                row = comparison._trial(receipt, trial, source=source)
                report = trial.get("supervisor") or {}
                checks["runtime_archive_bound_" + str(trial.get("trial"))] = (
                    row["runtime_archive_sha256"] == prepared.get("archive_sha256"))
                job = Path(config["jobs_dir"]) / config["job_name"]
                native_path = job / trial["trial"] / "result.json"
                checks["native_result_hash_bound_" + trial["trial"]] = (
                    native_path.is_file() and digest(native_path) == trial.get("native_result_sha256"))
                native = read(native_path) if native_path.is_file() else {}
                checks["native_reward_bound_" + trial["trial"]] = (
                    "verifier_result" in native
                    and trial.get("reward") == (native.get("verifier_result") or {}).get("rewards"))
                checks["native_exception_bound_" + trial["trial"]] = (
                    trial.get("exception_type") == (native.get("exception_info") or {}).get("exception_type"))
                checks["native_task_identity_bound_" + trial["trial"]] = (
                    trial.get("exact_trial_task_matches") is True
                    and native.get("task_name") == receipt["task"])
                supervisor_path = native_path.parent / "agent" / "supervisor-result.json"
                checks["supervisor_report_bound_" + trial["trial"]] = (
                    read(supervisor_path) == report if supervisor_path.is_file() else not report)
                checks["native_agent_context_bound_" + trial["trial"]] = (
                    trial.get("agent_context") == native.get("agent_result"))
                arithmetic = True
                native_calls = {call["invocation_id"]: call for call in report.get("provider_invocations", [])}
                for call in row["invocations"]:
                    values = call["known_token_subtotals"]
                    if call["native_usage_complete_observed"] is True and call["provider"] == "codex_cli":
                        arithmetic = arithmetic and (
                            all(type(values.get(key)) is int for key in comparison.TOKENS)
                            and values["total_tokens"] == values["input_tokens"] + values["output_tokens"]
                            and values["cached_input_tokens"] <= values["input_tokens"])
                        native_usage = (native_calls[call["invocation_id"]].get("native_rollout_usage") or {}).get("usage") or {}
                        reasoning = native_usage.get("reasoning_output_tokens")
                        if reasoning is not None:
                            arithmetic = arithmetic and (integer(reasoning) is not None
                                and integer(values["output_tokens"]) is not None
                                and reasoning <= values["output_tokens"])
                checks["native_usage_arithmetic_" + trial["trial"]] = arithmetic
                if not arithmetic:
                    row["tokens"] = dict.fromkeys(comparison.TOKENS)
                    row["native_usage_complete_observed"] = False
                row["comparison_label"] = label
                row["additional_native_usage_categories"] = extra_usage(report, row["invocations"])
                row["intent_preplanning_status"] = status_only(
                    (report.get("planning") or {}).get("intent_preplanning", report.get("intent_preplanning")))
                rows.append(row)
            outputs.append({"label": label, "source": source, "rows": rows,
                            "source384_selected": receipt.get("source384_enabled") is True})

    grouped = {label: comparison._aggregate([row for output in outputs if output["label"] == label
                                            for row in output["rows"]]) for label in expected_modes}
    empty_attempts = [output["source"] | {"label": output["label"]}
                      for output in outputs if not output["rows"]]
    for label, total in grouped.items():
        missing = sum(item["label"] == label for item in empty_attempts)
        total["retained_receipt_count"] = sum(item["label"] == label for item in outputs)
        total["receipts_without_native_trials"] = missing
        if missing:
            total["tokens"] = dict.fromkeys(comparison.TOKENS)
            total["provider_calls"] = total["reported_cost_usd"] = None
            efficiency = total["observed_efficiency"]
            efficiency["total_tokens_per_official_pass"] = None
            efficiency["reported_cost_usd_per_official_pass"] = None
            for key in ("token_ratio_unavailable_reasons", "cost_ratio_unavailable_reasons"):
                efficiency[key].append("retained_attempt_without_native_trial")
    qualified = all(all(row["checks"].values()) for row in controls)
    before = grouped["full-lexical-baseline"]["observed_efficiency"]["total_tokens_per_official_pass"]
    after = grouped["full-instruction-dedup"]["observed_efficiency"]["total_tokens_per_official_pass"]
    result = {
        "schema": "terminal-labeled-full-supervisor-pair@1",
        "report_source_sha256": digest(__file__),
        "comparison_source_sha256": digest(comparison.__file__),
        "native_usage_accounting_source_sha256": digest(native_usage_accounting.__file__),
        "controls": controls, "all_pair_checks_pass": qualified,
        "outputs": outputs, "label_totals": grouped,
        "receipts_without_trials": empty_attempts,
        "observed_token_ratio": after / before if qualified and before and after is not None else None,
        "native_codex_harness_compared": False,
        "causal_claim": False, "billing_total_verified": False,
        "unknown_fields_are_zero": False,
        "notes": [
            "Native receipt arms remain full; treatment labels are separate report keys.",
            "All supplied attempts are retained and charged; no retry is discarded.",
            "Counts cover complete native provider sessions, including planning and coding.",
            "Cache and reasoning categories are subsets, not additional total tokens.",
            "Setup, indexing, and nested phase durations are not added twice.",
            "A single paired task is descriptive and does not establish repeatable savings.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "all_pair_checks_pass": qualified,
                      "label_totals": grouped, "observed_token_ratio": result["observed_token_ratio"]}))


if __name__ == "__main__":
    main()
