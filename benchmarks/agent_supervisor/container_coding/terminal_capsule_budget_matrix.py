"""Register and prepare a bounded capsule-budget experiment without executing it.

Every planned slot is registered before preflight, including slots that fail or
never run. Existing single-trial preparation, execution and provider cooldown
remain the owners of runtime validation and dispatch.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import sys

from . import full_supervisor_benchmark as benchmark
from .benchmark_resource_profile import CODING600_SOURCE384_PROFILE, resource_environment
from .terminal_capsule_selection_policy import CAPSULE_BUDGETS
from .terminal_planning_policy import DECLARED_PLANNING_MODE

SCHEMA = "terminal-capsule-budget-matrix@1"
LABELS = dict(zip(CAPSULE_BUDGETS, ("legacy", "one", "two")))
PATH_OPTIONS = frozenset(("intent_requirement_contract", "intent_action_384_config",
                         "source384_config", "task_profile"))
FIXED_ARGUMENTS = frozenset(("dataset", "output", "archive", "arm", "task_name", "capsule_budget"))
PREPARATION_OPTIONS = frozenset(inspect.signature(benchmark.prepare).parameters) - FIXED_ARGUMENTS


def _wire(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_wire(value))


def prepare_matrix(*, dataset: Path, archive: Path, output: Path, task: str,
                   repeats: int = 3, options: dict | None = None) -> dict:
    """Prepare serially; execute argv is data and is never run by this function."""
    if type(repeats) is not int or not 1 <= repeats <= 10:
        raise ValueError("matrix repeats must be an integer from 1 to 10")
    if type(task) is not str or not task or len(task) > 128 or Path(task).name != task or task in {".", ".."}:
        raise ValueError("one dataset task name required")
    if options is not None and type(options) is not dict:
        raise ValueError("common preparation options must be an object")
    options = deepcopy(options or {})
    if set(options) - PREPARATION_OPTIONS:
        raise ValueError("unsupported or matrix-owned preparation option")
    # Explicit resource limits and a deterministic planning route by default.
    # Model-planning experiments use a separate matrix with that option changed.
    options.setdefault("resource_profile", CODING600_SOURCE384_PROFILE)
    options.setdefault("planning_mode", DECLARED_PLANNING_MODE)
    resource_environment(options["resource_profile"])
    for name in PATH_OPTIONS & options.keys():
        if options[name] is not None:
            options[name] = str(Path(options[name]).resolve(strict=True))
    _wire(options)  # Reject non-JSON options before creating an output directory.
    dataset, archive = Path(dataset).resolve(strict=True), Path(archive).resolve(strict=True)
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("matrix output must be a fresh canonical directory")
    slots = []
    for repeat in range(repeats):
        # Rotate which budget runs first. Three repeats balance every position;
        # this is a deterministic schedule, not a randomized causal experiment.
        order = CAPSULE_BUDGETS[repeat % 3:] + CAPSULE_BUDGETS[:repeat % 3]
        for budget in order:
            trial = output / f"repeat-{repeat + 1:02d}-{LABELS[budget]}"
            slots.append(dict(ordinal=len(slots) + 1, repeat=repeat + 1,
                capsule_budget=budget, output=str(trial), receipt=str(trial / "receipt.json"),
                execute_argv=[sys.executable, "-m", benchmark.__name__, "execute",
                              "--output", str(trial)]))
    collect = [sys.executable, "-m",
        "benchmarks.agent_supervisor.container_coding.terminal_capsule_budget_comparison"]
    for slot in slots:
        collect.extend(["--" + LABELS[slot["capsule_budget"]], slot["receipt"]])
    collect.extend(["--output", str(output / "comparison")])
    registry = dict(schema=SCHEMA, dataset=str(dataset), archive=str(archive), task=task,
        repeats=repeats, common_options=options, slots=slots, collect_argv=collect,
        schedule="rotating_budget_order", randomized=False, execution_concurrency=1,
        resource_limits=resource_environment(options["resource_profile"]),
        execution_automated=False, provider_calls=0, containers_launched=0,
        benchmark_advantage_claimed=False)
    output.mkdir(parents=True)
    _save(output / "matrix.json", registry)
    results = []
    for slot in slots:
        call_options = deepcopy(options)
        for name in PATH_OPTIONS & call_options.keys():
            if call_options[name] is not None:
                call_options[name] = Path(call_options[name])
        result = dict(ordinal=slot["ordinal"], capsule_budget=slot["capsule_budget"],
                      output=slot["output"], prepared=False)
        try:
            prepared = benchmark.prepare(dataset=dataset, archive=archive, output=Path(slot["output"]),
                task_name=task, arm="full", capsule_budget=slot["capsule_budget"], **call_options)
            result["prepared"] = prepared.get("prepared") is True
        except Exception as error:
            # Existing trial logs retain local diagnostics. The matrix does not
            # copy arbitrary exception text, task contents or credentials.
            result["error_type"] = type(error).__name__
        results.append(result)
        _save(output / f"preflight-{slot['ordinal']:02d}.json", result)
    summary = dict(schema=SCHEMA, registry_sha256=hashlib.sha256((output / "matrix.json").read_bytes()).hexdigest(),
        registered_trials=len(slots), prepared_trials=sum(row["prepared"] for row in results),
        all_prepared=all(row["prepared"] for row in results), preflights=results,
        provider_calls=0, containers_launched=0, trials_executed=0,
        benchmark_advantage_claimed=False)
    _save(output / "preflight.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--options", type=Path, help="JSON object of common full-supervisor prepare options")
    args = parser.parse_args(argv)
    options = json.loads(args.options.read_text()) if args.options else None
    result = prepare_matrix(dataset=args.dataset, archive=args.archive, output=args.output,
                            task=args.task, repeats=args.repeats, options=options)
    print(_wire(result), end="")
    return 0 if result["all_prepared"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
