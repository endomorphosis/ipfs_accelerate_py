"""Compare registered legacy/one/two capsule trials without launching providers.

Read each receipt beside its frozen preparation.json and config.json. The
experiment permits only capsule budget and job/output naming to differ. Native
usage keeps the existing collector's provider/cache semantics; prompt bytes and
stream snapshots never become token savings.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import stat

from . import benchmark_comparison as native
from . import benchmark_controls as controls
from .terminal_capsule_selection_policy import CAPSULE_BUDGETS, capsule_budget_limit, capsule_budget_observation

SCHEMA = "terminal-capsule-budget-comparison@1"
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
MAX_ATTEMPTS_PER_BUDGET = 16
MAX_INVOCATIONS = 1024
_BUDGET_DIFFERENCE = "experimental_controls_sha256.capsule_budget"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ASSETS = ("archive_sha256", "manifest_sha256", "host_source_sha256", "source384",
           "intent_action_384", "numerical_selection_384", "source384_enabled", "resource_profile")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate comparison input field")
        value[key] = item
    return value


def _read(path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None, None
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_RECEIPT_BYTES:
        raise ValueError("comparison input must be a bounded regular file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_RECEIPT_BYTES + 1)
    if len(raw) > MAX_RECEIPT_BYTES:
        raise ValueError("comparison input exceeds byte bound")
    value = json.loads(raw, object_pairs_hook=_object,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite comparison input")))
    if type(value) is not dict:
        raise ValueError("comparison input must be an object")
    return value, _sha(raw)


def _identity(value):
    return type(value) is str and 0 < len(value) <= 256


def _normalized_config(config):
    result = deepcopy(config)
    result.pop("job_name", None)
    result.pop("jobs_dir", None)
    result["agents"][0]["kwargs"].pop("capsule_budget", None)
    return result


def _frozen(path, budget, receipt):
    prepared, preparation_sha = _read(path.parent / "preparation.json")
    config, config_sha = _read(path.parent / "config.json")
    result = dict(preparation_sha256=preparation_sha, configuration_sha256=config_sha,
                  verified=False, reasons=[])
    if prepared is None or config is None:
        result["reasons"].append("missing_frozen_preparation_or_configuration")
        return result, None
    if (prepared.get("schema") != "terminal-full-supervisor-preparation@1"
            or prepared.get("arm") != "full" or type(config.get("agents")) is not list
            or len(config["agents"]) != 1 or type(config["agents"][0]) is not dict
            or type(config["agents"][0].get("kwargs")) is not dict):
        result["reasons"].append("invalid_frozen_preparation_or_configuration")
        return result, None
    kwargs = config["agents"][0]["kwargs"]
    declared = prepared.get("comparison_controls")
    if not controls.validate_controls(declared):
        result["reasons"].append("invalid_predeclared_controls")
        return result, None
    observed = controls.observe_controls(prepared, config,
        current_task_hashes=prepared.get("task_input_sha256"))
    if observed.get("status") != "observed" or prepared.get("config_sha256") != config_sha:
        result["reasons"].append("frozen_configuration_changed")
    if prepared.get("prepared") is not True:
        result["reasons"].append("preparation_not_successful")
    if (prepared.get("capsule_budget", "legacy") != budget
            or kwargs.get("capsule_budget", "legacy") != budget or kwargs.get("arm") != "full"):
        result["reasons"].append("frozen_budget_mismatch")
    identity = declared["identity"]
    if any(prepared.get(key) != value for key, value in identity.items()):
        result["reasons"].append("preparation_identity_mismatch")
    if receipt is not None:
        if receipt.get("comparison_controls") != observed:
            result["reasons"].append("receipt_controls_mismatch")
        if any(receipt.get(key) != value for key, value in identity.items()):
            result["reasons"].append("receipt_identity_mismatch")
        if receipt.get("original_task_inputs_unchanged") is not True:
            result["reasons"].append("original_task_inputs_unverified")
    if any(type(prepared.get(key)) is not str or not _DIGEST.fullmatch(prepared[key])
           for key in ("archive_sha256", "manifest_sha256")):
        result["reasons"].append("archive_binding_unavailable")
    from .benchmark_provider_profile import resolve_provider_profile
    provider = resolve_provider_profile(kwargs.get("provider_profile"))["provider"]
    result["verified"] = not result["reasons"]
    return result, dict(provider=provider, config=_normalized_config(config), controls=observed,
                       assets={key: prepared.get(key) for key in _ASSETS})


def _runtime_matches(receipt, trial, budget, frozen):
    reasons = []
    report = trial.get("supervisor")
    if type(report) is not dict:
        return ["runtime_report_unavailable"]
    if report.get("capsule_budget", "legacy") != budget:
        reasons.append("runtime_budget_mismatch")
    if budget != "legacy":
        expected = capsule_budget_observation(budget)
        if any(type(report.get(key)) is not type(value) or report.get(key) != value
               for key, value in expected.items()):
            reasons.append("runtime_budget_observation_unverified")
        if (receipt.get("capsule_budget_config_unchanged") is not True
                or receipt.get("capsule_budget_runtime_declared_matches") is not True):
            reasons.append("runtime_budget_join_unverified")
    if receipt.get("complete_single_trial_receipt") is not True or trial.get("exact_trial_task_matches") is not True:
        reasons.append("trial_task_join_unverified")
    if frozen is not None and any(call.get("provider") != frozen["provider"]
            or call.get("model") != receipt.get("model") for call in report.get("provider_invocations", [])):
        reasons.append("runtime_provider_or_model_mismatch")
    archive = native._archive_digest(trial)
    if frozen is None or archive is None or archive != frozen["assets"]["archive_sha256"]:
        reasons.append("runtime_archive_unverified")
    return reasons


def _repository_observation(report, frozen, binding):
    """Retain validated content-free telemetry, never infer absent retrieval."""
    from .terminal_repository_source_policy import (
        REPOSITORY_SOURCE_POLICIES, repository_source_dispatch_observation)

    selected = (frozen["config"]["agents"][0]["kwargs"].get("repository_source_policy", "legacy")
                if frozen is not None else None)
    result = dict(requested_policy=selected, status="unavailable", unavailable_reasons=[],
                  dispatch_observation=None)
    if frozen is None or not binding["verified"]:
        result["unavailable_reasons"].append("frozen_repository_policy_unverified")
    elif selected == "legacy":
        result["unavailable_reasons"].append("repository_source_policy_not_selected")
    elif selected not in REPOSITORY_SOURCE_POLICIES:
        result["unavailable_reasons"].append("unsupported_repository_source_policy")
    else:
        observation = repository_source_dispatch_observation(report, selected)
        result["dispatch_observation"] = observation
        for call in observation["invocations"]:
            if "retrieval_observation_valid" not in call:
                call.update(retrieval_observation_valid=None, retrieval_observation=None,
                            capsule_retrieval_observed=None)
        calls = observation["invocations"]
        if not calls or all(call["retrieval_observation_valid"] is None for call in calls):
            result["unavailable_reasons"].append("retained_retrieval_summary_unavailable")
        elif all(call["retrieval_observation_valid"] is True
                 and call["retrieval_observation"]["complete"] for call in calls):
            result["status"] = "observed"
        else:
            result["status"] = "partial"
            result["unavailable_reasons"].append("retained_retrieval_summary_incomplete_or_invalid")
    return result


def collect(*, legacy: list[Path], one: list[Path], two: list[Path]) -> dict:
    """Register every attempt explicitly, including paths to missing receipts.

    This is a posthoc readout, not proof of preregistration, actual runtime
    enforcement, exhaustive provider billing, or fixed generated plans.
    """
    requested = dict(zip(CAPSULE_BUDGETS, (legacy, one, two)))
    for paths in requested.values():
        if not isinstance(paths, (list, tuple)) or not 1 <= len(paths) <= MAX_ATTEMPTS_PER_BUDGET:
            raise ValueError("each capsule budget requires 1..16 registered receipt paths")
    attempts, rows, frozen_records = [], [], []
    seen_paths, seen_trials, seen_calls = set(), {}, set()
    for budget, paths in requested.items():
        for supplied in paths:
            path = Path(supplied).absolute()
            if path in seen_paths:
                raise ValueError("receipt path registered more than once")
            seen_paths.add(path)
            receipt, digest = _read(path)
            binding, frozen = _frozen(path, budget, receipt)
            attempt = dict(capsule_budget=budget, path=str(path), sha256=digest,
                status="missing_receipt" if receipt is None else "observed", frozen=binding,
                trial_count=0, distinct_trial_count=0, runtime_reasons=[])
            attempts.append(attempt)
            frozen_records.append(frozen)
            if receipt is None:
                continue
            if receipt.get("schema") != native.SUPERVISOR or receipt.get("arm") != "full":
                raise ValueError("capsule comparison requires full supervisor receipts")
            if receipt.get("capsule_budget", "legacy") != budget:
                raise ValueError("receipt registered under the wrong capsule budget")
            trials = receipt.get("trials")
            if (type(trials) is not list or len(trials) > 1 or type(receipt.get("trial_count")) is not int
                    or receipt["trial_count"] != len(trials)):
                raise ValueError("capsule comparison requires a single-trial receipt inventory")
            attempt["trial_count"] = len(trials)
            if not trials:
                attempt["status"] = "no_trial_observed"
            for trial in trials:
                if type(trial) is not dict or not _identity(trial.get("trial")):
                    raise ValueError("bounded native trial identity required")
                report = trial.get("supervisor") or {}
                if type(report) is not dict:
                    raise ValueError("invalid supervisor report")
                invocations = report.get("provider_invocations", [])
                if type(invocations) is not list or len(invocations) > MAX_INVOCATIONS:
                    raise ValueError("bounded provider invocation inventory required")
                if any(type(call) is not dict or not _identity(call.get("invocation_id")) for call in invocations):
                    raise ValueError("bounded provider invocation identities required")
                identity = (receipt.get("task"), trial["trial"])
                signature = _sha(_json(dict(budget=budget, trial=trial, controls=receipt.get("comparison_controls"),
                    joins={key: receipt.get(key) for key in ("complete_single_trial_receipt",
                        "capsule_budget_config_unchanged", "capsule_budget_runtime_declared_matches")})).encode())
                if identity in seen_trials:
                    if seen_trials[identity] != signature:
                        raise ValueError("conflicting or cross-budget native trial identity")
                    attempt["status"] = "duplicate_trial_observation"
                    continue
                seen_trials[identity] = signature
                call_ids = {(call.get("provider"), call["invocation_id"]) for call in invocations}
                if seen_calls & call_ids:
                    raise ValueError("provider invocation reused across distinct trials")
                seen_calls.update(call_ids)
                source = dict(path=str(path), sha256=digest, trial_count=len(trials))
                row = native._trial(receipt, trial, source=source)
                for call in row["invocations"]:
                    counts = call["tokens"]
                    if call["native_usage_complete_observed"] is True and call["cache_included_in_input"] is True:
                        values = [counts[name] for name in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")]
                        if (any(value is None for value in values) or values[0] + values[2] != values[3]
                                or values[1] > values[0]):
                            call["tokens"] = dict.fromkeys(native.TOKENS)
                            call["native_usage_complete_observed"] = None
                            row["tokens"] = dict.fromkeys(native.TOKENS)
                            row["native_usage_complete_observed"] = None
                            row["reported_cost_usd"] = None
                row["capsule_budget"] = budget
                row["maximum_capsules"] = capsule_budget_limit(budget)
                row["repository_source_observation"] = _repository_observation(report, frozen, binding)
                row["runtime_reasons"] = _runtime_matches(receipt, trial, budget, frozen)
                attempt["runtime_reasons"] = row["runtime_reasons"]
                selected = row["indexed_context_observation"].get("worker_capsules")
                row["selected_capsules_within_budget"] = type(selected) is int and 0 <= selected <= capsule_budget_limit(budget)
                if not row["selected_capsules_within_budget"]:
                    row["runtime_reasons"].append("selected_capsule_count_unverified")
                rows.append(row)
                attempt["distinct_trial_count"] += 1
    reference = next((value for attempt, value in zip(attempts, frozen_records)
                      if attempt["capsule_budget"] == "legacy" and attempt["frozen"]["verified"]), None)
    for attempt, frozen in zip(attempts, frozen_records):
        match = dict(matches=False, reasons=[], generic_control_comparison=None)
        if reference is None or frozen is None or not attempt["frozen"]["verified"]:
            match["reasons"].append("frozen_controls_unavailable_or_unverified")
        else:
            comparison = controls.compare_controls(reference["controls"], frozen["controls"])
            match["generic_control_comparison"] = comparison
            expected = [] if attempt["capsule_budget"] == "legacy" else [_BUDGET_DIFFERENCE]
            if comparison.get("differences") != expected:
                match["reasons"].append("unexpected_declared_control_difference")
            if reference["config"] != frozen["config"]:
                match["reasons"].append("configuration_differs_beyond_budget_and_output_names")
            if reference["assets"] != frozen["assets"]:
                match["reasons"].append("frozen_runtime_or_assets_differ")
            match["matches"] = not match["reasons"]
        attempt["experiment_control_comparison"] = match
    groups = {}
    for budget in CAPSULE_BUDGETS:
        selected = [row for row in rows if row["capsule_budget"] == budget]
        registrations = [attempt for attempt in attempts if attempt["capsule_budget"] == budget]
        aggregate = native._aggregate(selected)
        missing = sum(attempt["trial_count"] == 0 for attempt in registrations)
        complete = bool(selected) and missing == 0 and all(row["native_usage_complete_observed"] is True
            and row["tokens"]["total_tokens"] is not None for row in selected)
        if missing:
            # The generic aggregate only sees actual trials. Missing registered
            # attempts cannot silently turn its known subtotal into a full cost.
            aggregate["tokens"] = dict.fromkeys(native.TOKENS)
            aggregate["reported_cost_usd"] = None
            efficiency = aggregate["observed_efficiency"]
            for name in ("total_tokens_per_official_pass", "reported_cost_usd_per_official_pass"):
                efficiency[name] = None
            for name in ("token_ratio_unavailable_reasons", "cost_ratio_unavailable_reasons"):
                efficiency[name].append("registered_attempt_without_trial")
        groups[budget] = dict(aggregate=aggregate, registered_receipts=len(registrations),
            registered_attempts_without_trials=missing, complete_registered_native_usage=complete,
            experiment_controls_match=all(a["experiment_control_comparison"]["matches"] for a in registrations),
            runtime_joins_verified=bool(selected) and all(not row["runtime_reasons"] for row in selected),
            selected_capsule_counts=[row["indexed_context_observation"]["worker_capsules"] for row in selected],
            all_observed_trials_officially_passed=bool(selected) and all(row["official_reward"] == 1 for row in selected))
    equal_trials = len({g["aggregate"]["trials"] for g in groups.values()}) == 1
    controls_match = all(g["experiment_controls_match"] and g["runtime_joins_verified"] for g in groups.values())
    matched_success = controls_match and equal_trials and all(
        g["registered_attempts_without_trials"] == 0 and g["all_observed_trials_officially_passed"] for g in groups.values())
    contrasts = {}
    baseline = groups["legacy"]["aggregate"]["tokens"]["total_tokens"]
    for budget in CAPSULE_BUDGETS[1:]:
        reasons = []
        if not matched_success:
            reasons.append("matched_official_success_not_established")
        if not all(groups[name]["complete_registered_native_usage"] for name in ("legacy", budget)):
            reasons.append("incomplete_registered_native_usage")
        candidate = groups[budget]["aggregate"]["tokens"]["total_tokens"]
        if baseline is None or candidate is None:
            reasons.append("complete_native_total_unavailable")
        contrasts[budget] = dict(comparable=not reasons, unavailable_reasons=reasons,
            candidate_minus_legacy_tokens=candidate - baseline if not reasons else None,
            relative_token_change=(candidate / baseline - 1) if not reasons and baseline else None,
            causal_token_savings_claimed=False)
    return dict(schema=SCHEMA, attempts=attempts, rows=rows, budgets=groups, contrasts=contrasts,
        all_experiment_controls_match=controls_match, matched_official_success=matched_success,
        equal_distinct_trial_counts=equal_trials, provider_calls_by_collector=0,
        planning_outputs_held_fixed=False, preregistration_verified=False, runtime_enforcement_verified=False,
        exhaustive_pipeline_usage_verified=False, billing_total_verified=False, token_savings_claimed=False,
        collector_sha256=_sha(Path(__file__).read_bytes()),
        native_accounting_sha256=_sha(Path(native.__file__).read_bytes()),
        notes=["Only explicitly registered attempts are included; this report does not prove campaign completeness.",
               "All retained failed trials contribute usage; missing trials or incomplete native usage remain unknown.",
               "Prompt bytes and stream usage snapshots are observations, never additional tokens or savings.",
               "Budgets are ceilings; equal realized counts do not establish a changed capsule treatment.",
               "Retrieval telemetry describes prepared responses; absent historical summaries remain unknown, not zero or model attention.",
               "Model planning can change when capsule selection changes; a common planning mode is not a frozen plan.",
               "Differences are descriptive at matched official success, not causal evidence or verified billing."])


def markdown(result):
    lines = ["# Capsule budget comparison", "", "| Budget | Trials | Official passes | Complete native tokens | Selected capsules |",
             "| --- | ---: | ---: | ---: | --- |"]
    for budget, group in result["budgets"].items():
        totals = group["aggregate"]
        tokens = totals["tokens"]["total_tokens"]
        lines.append(f"| {budget} | {totals['trials']} | {totals['observed_official_passes']} | "
                     f"{tokens if tokens is not None else 'unknown'} | {group['selected_capsule_counts']} |")
    lines += ["", f"Matched official success: {result['matched_official_success']}.",
              "Missing attempts and partial usage remain unknown. Prompt bytes are not token savings.", ""]
    for budget, contrast in result["contrasts"].items():
        value = contrast["candidate_minus_legacy_tokens"]
        lines.append(f"- {budget}: candidate minus legacy native tokens: {value if value is not None else 'unavailable'}.")
    lines += ["", "These observations do not establish causal savings, complete pipeline cost, or verified billing.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("legacy", "one", "two"):
        parser.add_argument("--" + name, type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh exact comparison output directory required")
    result = collect(legacy=args.legacy, one=args.one, two=args.two)
    output.mkdir(parents=True)
    (output / "comparison.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    (output / "comparison.md").write_text(markdown(result))
    print(json.dumps(dict(schema=SCHEMA, matched_official_success=result["matched_official_success"],
                         provider_calls_by_collector=0)))


if __name__ == "__main__":
    main()
