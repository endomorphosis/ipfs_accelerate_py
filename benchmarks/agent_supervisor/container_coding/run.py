"""Offline container qualification, using the production conflict scheduler.

The scripted repair oracle is deliberately NOT an LLM or a supervisor daemon.
Solver receipts establish finite scheduling/state invariants, not code correctness.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import (
    materialize_task_conflict_graph,
)


TASKS = (
    {"task_id": "sum", "task_cid": "sum", "predicted_files": ["sum.py"],
     "expression": "sum(xs)", "cases": [([], 0), ([1, -2, 4], 3)], "deps": []},
    {"task_id": "unique", "task_cid": "unique", "predicted_files": ["unique.py"],
     "expression": "list(dict.fromkeys(xs))", "cases": [([], []), ([2, 1, 2], [2, 1])], "deps": []},
    {"task_id": "reverse", "task_cid": "reverse", "predicted_files": ["reverse.py"],
     "expression": "xs[::-1]", "cases": [([], []), ([1, 2, 3], [3, 2, 1])], "deps": []},
    {"task_id": "sum_followup", "task_cid": "sum_followup", "predicted_files": ["sum.py"],
     "expression": "sum(xs)", "cases": [([-4, -3], -7), ([9], 9)], "deps": ["sum"]},
)
MODES = {"serial": (1, False, False), "parallel": (3, False, False),
         "parallel_proved": (3, True, False), "parallel_proved_compact": (3, True, True)}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state_query(tasks, done, active, tested):
    """Encode a concrete snapshot; SAT means an invariant is violated.

    Names come from numeric indexes, never unescaped task text. The graph is
    checked against independently declared file surfaces and dependencies.
    """
    ids = [t["task_id"] for t in tasks]
    if len(set(ids)) != len(ids) or not (set(done) | set(active) | set(tested)) <= set(ids):
        raise ValueError("unknown or duplicate task identity")
    if len(active) != len(set(active)):
        raise ValueError("duplicate active lease")
    lines = ["(set-logic QF_UF)"]
    violations = []
    for i, task in enumerate(tasks):
        for prefix, population in (("d", done), ("a", active), ("t", tested)):
            lines += [f"(declare-const {prefix}{i} Bool)",
                      f"(assert (= {prefix}{i} {'true' if ids[i] in population else 'false'}))"]
        violations += [f"(and d{i} (not t{i}))", f"(and d{i} a{i})"]
        for dep in task["deps"]:
            j = ids.index(dep)
            violations.append(f"(and (or a{i} d{i}) (not d{j}))")
        for j in range(i):
            if set(task["predicted_files"]) & set(tasks[j]["predicted_files"]):
                violations.append(f"(and a{i} a{j})")
    lines += ["(assert (or " + " ".join(violations) + "))", "(check-sat)"]
    return "\n".join(lines) + "\n"


def prove(query, expected="unsat"):
    receipts = []
    for name, argv in (("z3", ["z3", "-in"]), ("cvc5", ["cvc5", "--lang=smt2"])):
        start = time.perf_counter()
        result = subprocess.run(argv, input=query, text=True, capture_output=True, timeout=10)
        verdict = result.stdout.strip()
        receipts.append({"solver": name, "verdict": verdict, "expected": expected,
                         "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                         "wall_seconds": time.perf_counter() - start})
        if result.returncode != 0 or verdict != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {verdict!r}: {result.stderr}")
    return receipts


def validate(workspace, task):
    path = workspace / task["predicted_files"][0]
    # The evaluator is supplied through stdin and is never placed in the workspace.
    code = ("import runpy\nf = runpy.run_path(" + repr(str(path)) + ")[\"solve\"]\n"
            + "\n".join(f"assert f({xs!r}) == {expected!r}" for xs, expected in task["cases"]))
    result = subprocess.run([sys.executable, "-"], input=code, text=True,
                            capture_output=True, timeout=10)
    return result.returncode == 0


def repair(workspace, task):
    start = time.perf_counter()
    (workspace / task["predicted_files"][0]).write_text(
        f"def solve(xs):\n    return {task['expression']}\n")
    passed = validate(workspace, task)
    return {"task_id": task["task_id"], "tests_passed": passed,
            "worker_seconds": time.perf_counter() - start,
            "provider_tokens": 0, "worker_kind": "scripted_oracle"}


def trial(mode, root):
    capacity, checked, compact = MODES[mode]
    workspace = root / mode
    workspace.mkdir()
    for task in TASKS:
        (workspace / task["predicted_files"][0]).write_text("def solve(xs):\n    return None\n")
    if any(validate(workspace, task) for task in TASKS):
        raise RuntimeError("negative control failed: broken seed passed")
    done, tested, events, receipts, rows, waves = set(), set(), [], [], [], []
    context_bytes = 0
    start = time.perf_counter()
    while len(done) < len(TASKS):
        ready = [t for t in TASKS if t["task_id"] not in done and set(t["deps"]) <= done]
        graph = materialize_task_conflict_graph(ready, max_lanes=capacity)
        if not graph.canonical_lanes:
            raise RuntimeError("no progress")
        # A color is a conflict-free wave; the graph's lanes aren't worker IDs.
        active = list(graph.canonical_lanes[0])
        batch = [t for t in ready if t["task_cid"] in active]
        if checked:
            receipts.extend(prove(state_query(TASKS, done, active, tested)))
        for task in batch:
            context = ({"task": task["task_id"], "files": task["predicted_files"],
                        "satisfied_dependencies": task["deps"]} if compact else
                       {"tasks": TASKS, "history": events, "current": task["task_id"]})
            context_bytes += len(json.dumps(context, sort_keys=True).encode())
        with ThreadPoolExecutor(max_workers=capacity) as pool:
            results = list(pool.map(lambda task: repair(workspace, task), batch))
        for row in results:
            rows.append(row)
            if not row["tests_passed"]:
                raise RuntimeError(f"repair failed: {row['task_id']}")
            tested.add(row["task_id"])
            done.add(row["task_id"])
            events.append({"task": row["task_id"], "status": "tested",
                           "previous": digest(events)})
        if checked:
            receipts.extend(prove(state_query(TASKS, done, [], tested)))
        waves.append(active)
    # Re-test the final shared tree, including tests for earlier tasks.
    final_passed = all(validate(workspace, task) for task in TASKS)
    return {"mode": mode, "passed": final_passed, "accepted_tasks": len(done) if final_passed else 0,
            "wall_seconds": time.perf_counter() - start, "waves": waves,
            "context_bytes": context_bytes, "provider_tokens": 0,
            "token_savings": None, "token_savings_status": "not_measured_no_model_calls",
            "proof_seconds": sum(r["wall_seconds"] for r in receipts),
            "proof_receipts": receipts, "workers": rows, "events": events,
            "final_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(workspace.glob("*.py"))}}


def run(repeats):
    if repeats < 1:
        raise ValueError("repeats must be positive")
    controls = {}
    for name, done, active, tested in (
        ("valid", [], ["sum", "unique"], []),
        ("unvalidated_completion", ["sum"], [], []),
        ("dependency_violation", [], ["sum_followup"], []),
        ("conflicting_writes", [], ["sum", "sum_followup"], []),
    ):
        controls[name] = prove(state_query(TASKS, done, active, tested),
                               "unsat" if name == "valid" else "sat")
    runs = []
    with tempfile.TemporaryDirectory(prefix="supervisor-coding-") as tmp:
        for repeat in range(repeats):
            root = Path(tmp) / str(repeat)
            root.mkdir()
            modes = list(MODES)
            # Rotate ordering to reduce consistently favoring warmed caches.
            modes = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            for mode in modes:
                runs.append({"repeat": repeat, **trial(mode, root)})
    import ipfs_accelerate_py.agent_supervisor.core.conflict_graph as scheduler
    return {"schema": "container-coding-qualification@1", "qualification_only": True,
            "full_supervisor_daemon_exercised": False, "worker_kind": "scripted_oracle",
            "corpus_sha256": digest(TASKS), "python": platform.python_version(),
            "scheduler_source_sha256": hashlib.sha256(Path(scheduler.__file__).read_bytes()).hexdigest(),
            "solver_versions": {s: subprocess.check_output([s, "--version"], text=True).splitlines()[0]
                                for s in ("z3", "cvc5")},
            "controls": controls, "runs": runs, "passed": all(r["passed"] for r in runs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = run(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "trials": len(report["runs"]),
                      "qualification_only": True, "provider_tokens": 0,
                      "output": str(args.output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
