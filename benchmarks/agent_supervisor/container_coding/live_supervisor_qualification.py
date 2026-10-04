"""Run an existing genuinely model-planned task through the admitted supervisor.

This is a bounded integration qualification, not a Terminal-Bench score. All
planning inputs and independent admission are reused without rewriting tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import verify_local_benchmark_admission
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def run(prepared: Path, output: Path, *, model="gpt-5.6-sol", timeout=90,
        independent_replay=False) -> dict:
    prepared, output = prepared.resolve(), output.resolve()
    if not 1 <= timeout <= 300:
        raise ValueError("coding timeout must be between 1 and 300 seconds")
    planner = json.loads((prepared / "result.json").read_text())
    if not planner.get("qualified") or not planner.get("live_planner_exercised"):
        raise ValueError("requires a genuine accepted model-generated planning receipt")
    admission = json.loads((prepared / "admission.json").read_text())
    verified = verify_local_benchmark_admission(admission, initial=True)
    graph = verified["graph"]
    if len(graph.tasks) != 1 or planner["task_cids"] != [graph.tasks[0].task_cid]:
        raise ValueError("qualification is bounded to the exact one planned task")
    task = graph.tasks[0]
    repository = prepared / "repository"
    bundle = json.loads((prepared / "context-nomination.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    database = prepared / "intent.duckdb"
    if independent_replay:
        from ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission import materialize_local_benchmark_plan
        from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
        database = output / "intent.duckdb"
        with IntentRepository(database) as intent:
            materialize_local_benchmark_plan(admission=admission, intent=intent)
    command = shlex.join([
        sys.executable, "-P", "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner",
        "--model", model, "--reasoning-effort", "high", "--timeout", str(timeout),
        "--max-output-tokens", "4096",
    ])
    report = {
        "schema": "admitted-live-supervisor-qualification@1", "qualified": False,
        "prepared": str(prepared), "task_cid": task.task_cid, "task_id": task.task_key,
        "context_bundle": bundle, "implementation_command": command,
        "model": model, "reasoning_effort": "high", "coding_timeout_seconds": timeout,
        "max_task_attempts": 1, "benchmark_performance_claimed": False,
        "independent_replay": independent_replay, "intent_database": str(database),
        "production_activation": False, "observations": [], "router_invocations": [],
    }
    started = time.monotonic()
    try:
        with open_existing_native_owner(
            database=database, checkout=repository,
            state_dir=output / "owner", repository_id=verified["manifest"]["repository_cid"],
            execution_routes={task.task_key: GROK_CODEX_EXECUTION_MODE},
        ) as owner:
            initial = owner.source.get_task(task.task_cid)
            if initial.status != "ready":
                raise ValueError("task must still be ready before its first coding attempt")
            runtime = AdmittedBenchmarkRuntime.create(
                output / "launch", admission=admission, server=owner.server, source=owner.source,
                implement=True, implementation_command=command, context_bundle=bundle,
                max_task_attempts=1, timeout_ms=30_000, lifetime_seconds=min(600, timeout + 180),
            )
            try:
                result = runtime.start()
                report["start"] = result.to_dict()
                if not result.succeeded:
                    raise RuntimeError(f"native START failed: {result.error}")
                deadline = time.monotonic() + timeout + 60
                previous = None
                while time.monotonic() < deadline:
                    current = owner.source.get_task(task.task_cid)
                    state = (current.status, current.revision)
                    if state != previous:
                        item = {"status": current.status, "revision": current.revision,
                                "elapsed_seconds": time.monotonic() - started}
                        report["observations"].append(item)
                        print(json.dumps(item), flush=True)
                        _write(output / "progress.json", report)
                        previous = state
                    if current.status in {"completed", "failed", "blocked", "cancelled"}:
                        break
                    heartbeat = runtime.state / "run/admitted_database_daemon_pass_heartbeat.json"
                    if heartbeat.is_file():
                        try:
                            last_pass = json.loads(heartbeat.read_text())
                        except (OSError, ValueError):
                            last_pass = {}
                        if last_pass.get("selection_idle_reason") == "expired_attempt_settlement_unavailable":
                            report["stopped_for_unavailable_settlement"] = last_pass
                            break
                    if not runtime.process.snapshot(runtime.profile).members:
                        raise RuntimeError("supervisor process tree ended before task completion")
                    time.sleep(.5)
                report["final_task"] = {"status": current.status, "revision": current.revision}
                report["observation"] = runtime.observe()
                report["bootstrap_receipts"] = runtime.bootstrap_receipts
                report["bootstrap_errors"] = runtime.bootstrap_errors
            finally:
                report["stop"] = runtime.stop().to_dict()
                report["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                runtime.close()
            report["qualified"] = (
                report["final_task"]["status"] == "completed"
                and report["stop"]["status"] == "succeeded"
                and report["remaining_processes"] == 0
                and not report["bootstrap_errors"]
            )
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2048]}
    finally:
        # Router receipts are emitted to native implementation logs even when
        # the provider fails. Keep each invocation once, never infer zeros.
        receipts = {}
        for path in output.rglob("*"):
            if not path.is_file() or path.suffix not in {".log", ".txt", ".out"}:
                continue
            if path.stat().st_size > 8_000_000:
                continue
            for line in path.read_text(errors="replace").splitlines():
                try:
                    value = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(value, dict) and value.get("schema") == "router-implementation-invocation@1":
                    receipts[value["invocation_id"]] = value
        report["router_invocations"] = list(receipts.values())
        report["usage_available"] = bool(receipts)
        report["seconds"] = time.monotonic() - started
        report["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        _write(output / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--independent-replay", action="store_true",
                        help="Materialize the same admitted graph into a new trial database; retain all previous attempts")
    args = parser.parse_args()
    result = run(args.prepared, args.output, model=args.model, timeout=args.timeout,
                 independent_replay=args.independent_replay)
    print(json.dumps({key: result[key] for key in ("qualified", "seconds", "router_invocations")}, sort_keys=True))
    return 0 if result["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
