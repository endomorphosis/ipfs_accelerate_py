"""Small coding experiment through llm_router; generated code runs only in Docker.

This exercises supervisor scheduling components, not the full daemon. Both API
and authenticated CLI providers are selected through the repository router.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from run import MODES, TASKS, digest, state_query, materialize_task_conflict_graph

API_PROVIDERS = frozenset({"openai", "openrouter", "xai", "meta_ai", "hf_inference_api"})
CLI_PROVIDERS = frozenset({"grok_cli", "codex_cli"})
PROVIDERS = API_PROVIDERS | CLI_PROVIDERS
INSTRUCTIONS = {
    "sum": "Implement solve(xs) returning the sum of a list of integers, including an empty list.",
    "unique": "Implement solve(xs) removing duplicate integers while preserving first occurrence order.",
    "reverse": "Implement solve(xs) returning the list of integers in reverse order.",
    "sum_followup": "Ensure solve(xs) sums negative integers and singleton lists correctly.",
}


def choose_route(provider=None, model=None):
    from ipfs_accelerate_py.llm_allocation.intelligence_index import (
        discover_available_providers, select_efficient_route,
    )
    discovered = sorted(set(discover_available_providers()))
    available = sorted(set(discovered) & PROVIDERS)
    if provider and provider not in PROVIDERS:
        raise ValueError("unsupported benchmark router provider")
    if provider and provider not in available:
        raise RuntimeError(f"requested router provider is unavailable: {provider}")
    if not available:
        raise RuntimeError(f"no supported router provider available; discovered: {discovered}")
    route = select_efficient_route(model_name=model or "auto", provider=provider or "",
                                   task_kind="coding", available_providers=available)
    if route.provider not in available or not route.model_name:
        raise RuntimeError("router did not select an available, explicit model")
    return {"provider": route.provider, "model": route.model_name,
            "reasoning_effort": route.reasoning_effort,
            "allocation_path": "cli" if route.provider in CLI_PROVIDERS else "api",
            "discovered_providers": discovered}


class MeteredProvider:
    """Retain native usage before the router's string API discards its envelope."""
    def __init__(self, inner, measurement):
        self.inner = inner
        self.measurement = measurement

    def generate(self, prompt, *, model_name=None, **kwargs):
        chat = getattr(self.inner, "chat_completions", None)
        if not callable(chat):
            return self.inner.generate(prompt, model_name=model_name, **kwargs)
        data = chat([{"role": "user", "content": prompt}], model_name=model_name, **kwargs)
        usage = data.get("usage", {})
        values = [usage.get("prompt_tokens"), usage.get("completion_tokens")]
        if all(type(v) is int and v >= 0 for v in values):
            self.measurement.update(provider_tokens=sum(values), input_tokens=values[0],
                                    output_tokens=values[1], usage_status="provider_reported")
        self.measurement["reported_model"] = data.get("model")
        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            raise ValueError("provider response has no text")
        return text


def generate(prompt, route, measurement, *, generate_fn=None, provider_instance=None):
    if route["provider"] not in PROVIDERS:
        raise ValueError("unsupported benchmark router provider")
    from ipfs_accelerate_py.router_deps import RouterDeps
    deps = RouterDeps()  # No response cache shared across experimental arms.
    if generate_fn is None:
        from ipfs_accelerate_py.llm_router import generate_text, get_llm_provider
        generate_fn = generate_text
        provider_instance = get_llm_provider(route["provider"], deps=deps, use_cache=False)
    metered = MeteredProvider(provider_instance, measurement)
    cli = route["provider"] in CLI_PROVIDERS
    if cli:
        from ipfs_accelerate_py.cli_runtime.cli_metadata import (
            set_last_cli_observation, get_last_cli_observation,
        )
        # Seed this thread so timeouts cannot reuse another call's observations.
        set_last_cli_observation(route["provider"], {})
    options = {"reasoning_effort": route["reasoning_effort"]} if route.get("reasoning_effort") else {}
    try:
        return generate_fn(prompt, provider=route["provider"], model_name=route["model"],
                       provider_instance=metered, deps=deps,
                       allow_local_fallback=False, allow_cross_provider_fallback=False,
                       max_tokens=512, max_new_tokens=512, temperature=0,
                       timeout=90, task_kind="coding", allocation_path="cli" if cli else "api",
                       allocation_session_id=str(uuid.uuid4()), **options)
    finally:
        if cli:
            obs = get_last_cli_observation(route["provider"])
            values = [obs.get("prompt_tokens"), obs.get("completion_tokens")]
            if all(type(v) is int and v >= 0 for v in values):
                measurement.update(provider_tokens=sum(values), input_tokens=values[0],
                                   output_tokens=values[1], usage_status="provider_reported_cli")
            for key in ("cached_tokens", "total_cost_usd", "session_id", "model_id", "num_turns"):
                if key in obs:
                    measurement[key] = obs[key]


# Fixed RPC code. Model output is JSON data, never host Python or shell text.
RPC = '''import json, sys
from pathlib import Path
from run import TASKS, validate, prove
p = json.load(sys.stdin)
if p["op"] == "prove":
    result = prove(p["query"])
else:
    w = Path("/tmp/work")
    w.mkdir(exist_ok=True)
    t = TASKS[p["index"]]
    f = w / t["predicted_files"][0]
    if p["op"] == "write":
        f.write_text(p["code"])
        result = True
    elif p["op"] == "read":
        result = f.read_text()
    else:
        result = validate(w, t)
print(json.dumps(result))
'''


class Sandbox:
    def __init__(self, image):
        self.image = image
        self.name = "coding-bench-" + uuid.uuid4().hex

    def __enter__(self):
        repo = Path(__file__).resolve().parents[3]
        subprocess.run(["docker", "run", "-d", "--name", self.name,
                        "--network", "none", "--read-only", "--cap-drop", "ALL",
                        "--security-opt", "no-new-privileges", "--pids-limit", "128",
                        "--memory", "1g", "--cpus", "3", "--tmpfs", "/tmp:rw,nosuid,size=128m",
                        "--mount", f"type=bind,src={repo},dst=/source,readonly",
                        "--entrypoint", "sleep", self.image, "3600"],
                       check=True, capture_output=True, text=True, timeout=30)
        return self

    def call(self, op, **payload):
        result = subprocess.run(["docker", "exec", "-i", self.name, "python", "-c", RPC],
                                input=json.dumps({"op": op, **payload}), text=True,
                                capture_output=True, timeout=30)
        if result.returncode:
            # Avoid persisting arbitrary model-generated stderr in host logs.
            raise RuntimeError("container operation failed: " + op)
        return json.loads(result.stdout)

    def __exit__(self, *exc):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, timeout=30)


def trial(mode, route, image):
    capacity, checked, compact = MODES[mode]
    rows, proofs, waves, history = [], [], [], []
    done, tested, attempted = set(), set(), set()
    with Sandbox(image) as sandbox:
        for index in range(len(TASKS)):
            sandbox.call("write", index=index, code="def solve(xs):\n    return None\n")
        if any(sandbox.call("validate", index=i) for i in range(len(TASKS))):
            raise RuntimeError("broken-seed negative control passed")
        started = time.perf_counter()
        while True:
            ready = [t for t in TASKS if t["task_id"] not in attempted and set(t["deps"]) <= done]
            if not ready:
                break
            graph = materialize_task_conflict_graph(ready, max_lanes=capacity)
            active = list(graph.canonical_lanes[0])
            batch = [t for t in ready if t["task_id"] in active]
            if checked:
                proofs.extend(sandbox.call("prove", query=state_query(TASKS, done, active, tested)))

            def worker(task):
                index = next(i for i, t in enumerate(TASKS) if t["task_id"] == task["task_id"])
                context = {"instruction": INSTRUCTIONS[task["task_id"]],
                           "source": sandbox.call("read", index=index)}
                if compact:
                    context["satisfied_dependencies"] = task["deps"]
                else:
                    context.update(instructions=INSTRUCTIONS, history=history)
                prompt = ("Repair this Python function. Return only a JSON object with one key, code, "
                          "containing the complete source. No markdown or tools.\n" + json.dumps(context))
                begin = time.perf_counter()
                row = {"task_id": task["task_id"], "prompt_bytes": len(prompt.encode()),
                       "provider_tokens": None, "usage_status": "unavailable_router_text_api",
                       "tests_passed": False}
                try:
                    response = str(generate(prompt, route, row))
                    row["response_bytes"] = len(response.encode())
                    payload = json.loads(response)
                    code = payload["code"]
                    if not isinstance(code, str) or len(code.encode()) > 32768:
                        raise ValueError("invalid code payload")
                    sandbox.call("write", index=index, code=code)
                    row["tests_passed"] = sandbox.call("validate", index=index)
                    row["source_sha256"] = digest(code)
                except Exception as exc:
                    row["error_type"] = type(exc).__name__
                row["worker_seconds"] = time.perf_counter() - begin
                return row

            with ThreadPoolExecutor(max_workers=capacity) as pool:
                results = list(pool.map(worker, batch))
            for row in results:
                rows.append(row)
                attempted.add(row["task_id"])
                if row["tests_passed"]:
                    done.add(row["task_id"])
                    tested.add(row["task_id"])
                history.append({"task": row["task_id"], "tests_passed": row["tests_passed"]})
            if checked:
                proofs.extend(sandbox.call("prove", query=state_query(TASKS, done, [], tested)))
            waves.append(active)
        final_tests = [sandbox.call("validate", index=i) for i in range(len(TASKS))]
        tokens = (sum(r["provider_tokens"] for r in rows)
                  if rows and all(r["provider_tokens"] is not None for r in rows) else None)
        return {"mode": mode, "route": route, "calls": len(rows), "workers": rows,
                "waves": waves, "proof_receipts": proofs, "final_tests": final_tests,
                "passed": len(done) == len(TASKS) and all(final_tests),
                "wall_seconds": time.perf_counter() - started,
                "provider_tokens": tokens, "token_savings": None,
                "blocked_tasks": sorted(set(t["task_id"] for t in TASKS) - attempted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(PROVIDERS))
    parser.add_argument("--model")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--image", default="ipfs-supervisor-coding-qualification:local")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "container-coding-router-pilot@1", "full_supervisor_daemon_exercised": False,
              "max_router_calls": 16, "requested_max_output_tokens_per_call": 512,
              "corpus_sha256": digest(TASKS), "runs": [], "token_savings": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        report["route"] = choose_route(args.provider, args.model)
        if args.preflight:
            report["status"] = "ready"
        else:
            # Freeze image identity once across all arms.
            image = subprocess.check_output(["docker", "image", "inspect", "--format",
                                             "{{.Id}}", args.image], text=True).strip()
            report["image_id"] = image
            for mode in MODES:
                report["runs"].append(trial(mode, report["route"], image))
                save()
            baseline = report["runs"][0]
            for row in report["runs"]:
                if (baseline["passed"] and row["passed"] and baseline["provider_tokens"]
                        and row["provider_tokens"] is not None):
                    row["token_savings"] = 1 - row["provider_tokens"] / baseline["provider_tokens"]
            report["status"] = "completed" if all(r["passed"] for r in report["runs"]) else "failed"
    except Exception as exc:
        report["status"] = "blocked"
        report["error_type"] = type(exc).__name__
        # Preflight errors originate locally and contain no provider responses.
        if not report.get("route"):
            report["reason"] = str(exc)
    save()
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 2 if report["status"] == "blocked" else (1 if report["status"] == "failed" else 0)


if __name__ == "__main__":
    raise SystemExit(main())
